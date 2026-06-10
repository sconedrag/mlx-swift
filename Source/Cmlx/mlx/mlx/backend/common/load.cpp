// Copyright © 2023 Apple Inc.

#include <algorithm>
#include <utility>

#include "mlx/primitives.h"
#include "mlx/scheduler.h"

#if defined(__APPLE__)
#include <mach/vm_page_size.h>            // vm_page_size (page-alignment check)
#include "mlx/backend/metal/allocator.h" // metal::allocator() + malloc_nocopy
#include "mlx/io/load.h"                  // io::MmapFileReader (+ <cstring>)
#endif

namespace {

template <const uint8_t scalar_size>
void swap_endianness(uint8_t* data_bytes, size_t N) {
  struct Elem {
    uint8_t bytes[scalar_size];
  };

  Elem* data = reinterpret_cast<Elem*>(data_bytes);

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < (scalar_size / 2); j++) {
      std::swap(data[i].bytes[j], data[i].bytes[scalar_size - j - 1]);
    }
  }
}

} // namespace

namespace mlx::core {

void Load::eval_cpu(const std::vector<array>& inputs, array& out) {
#if defined(__APPLE__)
  // W4 (mmap weight serving): if the reader is an MmapFileReader, build the
  // output's Metal buffer directly over the mmap'd file region (zero-copy,
  // clean/file-backed pages) instead of malloc + read. Page-aligned tensors use
  // newBufferWithBytesNoCopy; unaligned tensors fall back to malloc + memcpy
  // from the mapped region (Study 12 RQ3: most safetensors offsets are NOT
  // 16 KB-aligned, so the fallback is the common case for individual tensors).
  auto mmap_reader = dynamic_cast<io::MmapFileReader*>(reader_.get());
  if (mmap_reader) {
    const char* data_ptr = mmap_reader->data_at(offset_);
    if (!data_ptr) {
      throw std::runtime_error(
          "[Load::eval_cpu] mmap data_at returned null for offset " +
          std::to_string(offset_));
    }

    size_t data_size = out.size() * out.itemsize();

    uintptr_t addr = reinterpret_cast<uintptr_t>(data_ptr);
    if (addr % vm_page_size != 0) {
      // Not page-aligned: standard malloc + memcpy from the mapped region.
      out.set_data(allocator::malloc(out.nbytes()));
      std::memcpy(out.data<char>(), data_ptr, data_size);
    } else {
      // Page-aligned: zero-copy Metal buffer over the mmap'd pages.
      auto& metal_alloc = metal::allocator();
      auto buf = metal_alloc.malloc_nocopy(const_cast<char*>(data_ptr), data_size);
      out.set_data(buf);
    }

    // Apple Silicon is little-endian and safetensors is little-endian — no swap.
    // Route through the scheduler to preserve stream ordering with downstream ops.
    scheduler::enqueue(stream(), []() { /* data already available via mmap */ });
  } else
#endif
  {
    // STANDARD PATH: pread-based loading (unchanged).
    out.set_data(allocator::malloc(out.nbytes()));
    auto read_task = [out_ptr = out.data<char>(),
                      size = out.size(),
                      itemsize = out.itemsize(),
                      offset = offset_,
                      reader = reader_,
                      swap_endianness_ = swap_endianness_]() mutable {
      reader->read(out_ptr, size * itemsize, offset);
      if (swap_endianness_) {
        switch (itemsize) {
          case 2:
            swap_endianness<2>(reinterpret_cast<uint8_t*>(out_ptr), size);
            break;
          case 4:
            swap_endianness<4>(reinterpret_cast<uint8_t*>(out_ptr), size);
            break;
          case 8:
            swap_endianness<8>(reinterpret_cast<uint8_t*>(out_ptr), size);
            break;
        }
      }
    };
    auto fut = io::thread_pool().enqueue(std::move(read_task)).share();
    scheduler::enqueue(stream(), [fut = std::move(fut)]() { fut.wait(); });
  }
}

} // namespace mlx::core
