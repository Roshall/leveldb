// Copyright (c) 2017 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#ifndef STORAGE_LEVELDB_HOT_INCLUDE_EXPORT_H_
#define STORAGE_LEVELDB_HOT_INCLUDE_EXPORT_H_

#if !defined(LEVELDB_HOT_EXPORT)

#if defined(LEVELDB_HOT_SHARED_LIBRARY)
#if defined(_WIN32)

#if defined(LEVELDB_HOT_COMPILE_LIBRARY)
#define LEVELDB_HOT_EXPORT __declspec(dllexport)
#else
#define LEVELDB_HOT_EXPORT __declspec(dllimport)
#endif  // defined(LEVELDB_HOT_COMPILE_LIBRARY)

#else  // defined(_WIN32)
#if defined(LEVELDB_HOT_COMPILE_LIBRARY)
#define LEVELDB_HOT_EXPORT __attribute__((visibility("default")))
#else
#define LEVELDB_HOT_EXPORT
#endif
#endif  // defined(_WIN32)

#else  // defined(LEVELDB_HOT_SHARED_LIBRARY)
#define LEVELDB_HOT_EXPORT
#endif

#endif  // !defined(LEVELDB_HOT_EXPORT)

#endif  // STORAGE_LEVELDB_HOT_INCLUDE_EXPORT_H_
