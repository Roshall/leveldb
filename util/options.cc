// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "leveldb_hot/options.h"

#include "leveldb_hot/comparator.h"
#include "leveldb_hot/env.h"

namespace leveldb_hot {

Options::Options() : comparator(BytewiseComparator()), env(Env::Default()) {}

}  // namespace leveldb_hot
