/*
   Portions Copyright (c) 2016-Present, Facebook, Inc.
   Portions Copyright (c) 2014, SkySQL Ab

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; version 2 of the License.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA */

#pragma once

/* C++ system header files */
#include <map>
#include <string>
#include <string_view>
#include <vector>

/* RocksDB header files */
#include "rocksdb/db.h"

/* MyRocks header files */
#include "./rdb_cf_options.h"
#include "./rdb_datadic.h"

namespace myrocks {

/*
  We need a Column Family (CF) manager. Its functions:
  - create column families (synchronized, don't create the same twice)
  - keep count in each column family.
     = the count is kept on-disk.
     = there are no empty CFs. initially count=1.
     = then, when doing DDL, we increase or decrease it.
       (atomicity is maintained by being in the same WriteBatch with DDLs)
     = if DROP discovers that now count=0, it removes the CF.

  Current state is:
  - CFs are created in a synchronized way. We can't remove them, yet.
*/

class Rdb_cf_manager : public Ensure_initialized {
  std::map<std::string, rocksdb::ColumnFamilyHandle *> m_cf_name_map;
  std::map<uint32_t, rocksdb::ColumnFamilyHandle *> m_cf_id_map;

  mutable mysql_mutex_t m_mutex;

  std::unique_ptr<Rdb_cf_options> m_cf_options = nullptr;

  uint32_t tmp_column_family_id;
  uint32_t tmp_system_column_family_id;
  rocksdb::DB *m_db = nullptr;

 public:
  Rdb_cf_manager(const Rdb_cf_manager &) = delete;
  Rdb_cf_manager &operator=(const Rdb_cf_manager &) = delete;
  Rdb_cf_manager() = default;

  [[nodiscard]] static bool is_cf_name_reverse(std::string_view name);

  /*
    This is called right after the DB::Open() call. The parameters describe
    column
    families that are present in the database. The first CF is the default CF.

    @param db [IN]: rocksdb transaction
    @param cf_options [IN]: properties of column families.
    @param handles [IN][OUT]: list of all active cf_handles fetched from rdb
    transaction.
  */
  bool init(rocksdb::DB *const db, std::unique_ptr<Rdb_cf_options> &&cf_options,
            std::vector<rocksdb::ColumnFamilyHandle *> *handles);
  void cleanup();

  /*
    Used by CREATE TABLE.
    cf_name requires non-empty string
  */
  rocksdb::ColumnFamilyHandle *get_or_create_cf(const std::string &cf_name);

  /* Used by table open */
  rocksdb::ColumnFamilyHandle *get_cf(const std::string &cf_name) const;

  /* Look up cf by id; used by datadic */
  rocksdb::ColumnFamilyHandle *get_cf(const uint32_t id) const;

  /* Used to iterate over column families for show status */
  std::vector<std::string> get_cf_names(void) const;

  /* Used to iterate over column families */
  std::vector<rocksdb::ColumnFamilyHandle *> get_all_cf(void) const;

  int remove_dropped_cf(Rdb_dict_manager *const dict_manager,
                        const uint32 &cf_id);

  /* Used to delete cf by name */
  int drop_cf(Rdb_ddl_manager *const ddl_manager,
              Rdb_dict_manager *const dict_manager, const std::string &cf_name);

  /* Create cf flags if it does not exist */
  [[nodiscard]] int create_cf_flags_if_needed(
      const Rdb_dict_manager &dict_manager, uint32_t cf_id,
      std::string_view cf_name, bool is_per_partition_cf = false);

  /* return true when success */
  bool get_cf_options(const std::string &cf_name,
                      rocksdb::ColumnFamilyOptions *const opts)
      MY_ATTRIBUTE((__nonnull__)) {
    return m_cf_options->get_cf_options(cf_name, opts);
  }

  void update_options_map(const std::string &cf_name,
                          const std::string &updated_options) {
    m_cf_options->update(cf_name, updated_options);
  }

  bool is_tmp_column_family(const uint cf_id) const;

  /* Used by bulk load */
  void drop_cf_from_map(const std::string &cf_name, const uint32_t cf_id);

 private:
  rocksdb::ColumnFamilyHandle *get_cf(const std::string &cf_name,
                                      const bool lock_held_by_caller) const;
};

}  // namespace myrocks
