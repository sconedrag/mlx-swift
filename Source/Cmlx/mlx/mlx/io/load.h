// Copyright © 2023 Apple Inc.

#pragma once

#include <memory>
#include <sstream>

#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#ifdef _MSC_VER
#include <io.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "mlx/threadpool.h"

// Strictly we need to operate on files in binary mode (to avoid \r getting
// automatically inserted), but every modern system except for Windows no
// longer differentiates between binary and text files and for them define
// the flag as no-op.
#ifndef O_BINARY
#define O_BINARY 0
#endif

namespace mlx::core {

namespace io {

ThreadPool& thread_pool();

class Reader {
 public:
  virtual bool is_open() const = 0;
  virtual bool good() const = 0;
  virtual size_t tell() = 0; // tellp is non-const in iostream
  virtual void seek(
      int64_t off,
      std::ios_base::seekdir way = std::ios_base::beg) = 0;
  virtual void read(char* data, size_t n) = 0;
  virtual void read(char* data, size_t n, size_t offset) = 0;
  virtual std::string label() const = 0;
  virtual ~Reader() = default;
};

class Writer {
 public:
  virtual bool is_open() const = 0;
  virtual bool good() const = 0;
  virtual size_t tell() = 0;
  virtual void seek(
      int64_t off,
      std::ios_base::seekdir way = std::ios_base::beg) = 0;
  virtual void write(const char* data, size_t n) = 0;
  virtual std::string label() const = 0;
  virtual ~Writer() = default;
};

class ParallelFileReader : public Reader {
 public:
  explicit ParallelFileReader(std::string file_path)
      : fd_(open(file_path.c_str(), O_RDONLY | O_BINARY)),
        label_(std::move(file_path)) {}

  ~ParallelFileReader() override {
    close(fd_);
  }

  bool is_open() const override {
    return fd_ > 0;
  }

  bool good() const override {
    return is_open();
  }

  size_t tell() override {
    return lseek(fd_, 0, SEEK_CUR);
  }

  // Warning: do not use this function from multiple threads as
  // it advances the file descriptor
  void seek(int64_t off, std::ios_base::seekdir way = std::ios_base::beg)
      override {
    if (way == std::ios_base::beg) {
      lseek(fd_, off, 0);
    } else {
      lseek(fd_, off, SEEK_CUR);
    }
  }

  // Warning: do not use this function from multiple threads as
  // it advances the file descriptor
  void read(char* data, size_t n) override;

  void read(char* data, size_t n, size_t offset) override;

  std::string label() const override {
    return "file " + label_;
  }

 private:
  static constexpr size_t batch_size_ = 1 << 25;
  static ThreadPool& thread_pool();
  int fd_;
  std::string label_;
};

// W4 (mmap weight serving): a Reader that memory-maps the file (MAP_PRIVATE,
// PROT_READ|PROT_WRITE for COW safety) so weight pages are file-backed/clean.
// read() still memcpy's (used for header parsing); the zero-copy win is in
// Load::eval_cpu, which calls data_at() + the allocator's malloc_nocopy to
// build Metal buffers directly over the mapped region. Ported from the
// sconedrag/mlx-swift mmap-weight-loading fork (Study 12) onto mlx-core 0.30.6.
class MmapFileReader : public Reader {
 public:
  explicit MmapFileReader(std::string file_path)
      : label_(std::move(file_path)) {
    fd_ = open(label_.c_str(), O_RDONLY);
    if (fd_ < 0) return;

    struct stat st;
    if (fstat(fd_, &st) < 0) {
      close(fd_);
      fd_ = -1;
      return;
    }
    file_size_ = st.st_size;

    // MAP_PRIVATE means writes go to COW pages, not the file.
    // PROT_WRITE prevents SIGBUS if Metal accidentally writes to a weight page.
    // In practice, LoRALinear never writes to base weights, so no COW occurs.
    mapped_ = static_cast<char*>(
        mmap(nullptr, file_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd_, 0));
    if (mapped_ == MAP_FAILED) {
      mapped_ = nullptr;
      close(fd_);
      fd_ = -1;
      return;
    }

    // Advise sequential access for initial header parsing
    posix_madvise(mapped_, file_size_, POSIX_MADV_SEQUENTIAL);
  }

  ~MmapFileReader() override {
    if (mapped_) {
      munmap(mapped_, file_size_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  MmapFileReader(const MmapFileReader&) = delete;
  MmapFileReader& operator=(const MmapFileReader&) = delete;

  bool is_open() const override { return mapped_ != nullptr; }
  bool good() const override { return is_open(); }

  size_t tell() override { return cursor_; }

  void seek(int64_t off, std::ios_base::seekdir way = std::ios_base::beg)
      override {
    if (way == std::ios_base::beg) {
      cursor_ = off;
    } else if (way == std::ios_base::end) {
      cursor_ = file_size_ + off; // seek(0, end) -> cursor = file_size
    } else {
      cursor_ += off;
    }
  }

  void read(char* data, size_t n) override {
    if (cursor_ + n > file_size_) {
      throw std::runtime_error(
          "[MmapFileReader] Read past end of file: " + label_);
    }
    std::memcpy(data, mapped_ + cursor_, n);
    cursor_ += n;
  }

  void read(char* data, size_t n, size_t offset) override {
    if (offset + n > file_size_) {
      throw std::runtime_error(
          "[MmapFileReader] Read past end of file: " + label_);
    }
    std::memcpy(data, mapped_ + offset, n);
  }

  std::string label() const override { return "mmap " + label_; }

  /// Get raw pointer to mmap'd data at offset (for zero-copy buffer creation)
  const char* data_at(size_t offset) const {
    if (offset >= file_size_) return nullptr;
    return mapped_ + offset;
  }

  size_t file_size() const { return file_size_; }

  /// Pre-warm a range of pages via madvise(WILLNEED)
  void prewarm(size_t offset, size_t length) const {
    if (!mapped_) return;
    size_t end = std::min(offset + length, file_size_);
    posix_madvise(mapped_ + offset, end - offset, POSIX_MADV_WILLNEED);
  }

 private:
  int fd_ = -1;
  char* mapped_ = nullptr;
  size_t file_size_ = 0;
  size_t cursor_ = 0;
  std::string label_;
};

class FileWriter : public Writer {
 public:
  explicit FileWriter() {}
  explicit FileWriter(std::string file_path)
      : fd_(open(
            file_path.c_str(),
            O_CREAT | O_WRONLY | O_TRUNC | O_BINARY,
            0644)),
        label_(std::move(file_path)) {}

  FileWriter(const FileWriter&) = delete;
  FileWriter& operator=(const FileWriter&) = delete;
  FileWriter(FileWriter&& other) {
    std::swap(fd_, other.fd_);
  }

  ~FileWriter() override {
    if (fd_ != 0) {
      close(fd_);
    }
  }

  bool is_open() const override {
    return fd_ >= 0;
  }

  bool good() const override {
    return is_open();
  }

  size_t tell() override {
    return lseek(fd_, 0, SEEK_CUR);
  }

  void seek(int64_t off, std::ios_base::seekdir way = std::ios_base::beg)
      override {
    if (way == std::ios_base::beg) {
      lseek(fd_, off, 0);
    } else {
      lseek(fd_, off, SEEK_CUR);
    }
  }

  void write(const char* data, size_t n) override {
    while (n != 0) {
      auto m = ::write(fd_, data, std::min(n, static_cast<size_t>(INT32_MAX)));
      if (m <= 0) {
        std::ostringstream msg;
        msg << "[write] Unable to write " << n << " bytes to file.";
        throw std::runtime_error(msg.str());
      }
      data += m;
      n -= m;
    }
  }

  std::string label() const override {
    return "file " + label_;
  }

 private:
  int fd_{0};
  std::string label_;
};

} // namespace io
} // namespace mlx::core
