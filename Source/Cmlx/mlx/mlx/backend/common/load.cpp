// Copyright © 2023 Apple Inc.

#include <algorithm>
#include <utility>

#include "mlx/primitives.h"
#include "mlx/scheduler.h"

#if defined(__APPLE__)
#include "mlx/backend/metal/allocator.h"  // for malloc_nocopy
#include "mlx/io/load.h"                  // for MmapFileReader
#include <mach/vm_page_size.h>            // for vm_page_size (page alignment check)
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
  // Check if the reader is an MmapFileReader for zero-copy path
  auto mmap_reader = dynamic_cast<io::MmapFileReader*>(reader_.get());

  if (mmap_reader) {
    // MMAP PATH: Create a Metal buffer that references the mmap'd file data
    // directly. No copy, no allocation. OS manages paging from NVMe.
    const char* data_ptr = mmap_reader->data_at(offset_);
    if (!data_ptr) {
      throw std::runtime_error("[Load::eval_cpu] mmap data_at returned null for offset " +
                               std::to_string(offset_));
    }

    size_t data_size = out.size() * out.itemsize();

    // Check page alignment for newBufferWithBytesNoCopy
    uintptr_t addr = reinterpret_cast<uintptr_t>(data_ptr);
    if (addr % vm_page_size != 0) {
      // Tensor data is not page-aligned within the safetensors file.
      // Fall back to standard malloc + memcpy from mmap'd region.
      out.set_data(allocator::malloc(out.nbytes()));
      std::memcpy(out.data<char>(), data_ptr, data_size);
    } else {
      // Page-aligned: use zero-copy Metal buffer
      auto& metal_alloc = metal::allocator();
      auto buf = metal_alloc.malloc_nocopy(
          const_cast<char*>(data_ptr), data_size);
      out.set_data(buf);
    }

    // Endianness swap: NOT NEEDED on Apple Silicon (always little-endian).
    // Safetensors format is little-endian. ARM is little-endian. No swap required.
    // The swap_endianness_ flag is always false for ARM targets.

    // Route through scheduler to maintain compute graph ordering.
    // Even though the data is already accessible (mmap'd), MLX's stream
    // ordering requires this to prevent race conditions with downstream ops.
    scheduler::enqueue(stream(), []() { /* data already available via mmap */ });
  } else
#endif
  {
    // STANDARD PATH: Original pread-based loading (unchanged)
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
