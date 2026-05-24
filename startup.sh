#!/bin/bash
set -e

PG_BIN="/usr/lib/postgresql/15/bin"
PG_DATA="/data/postgres"

# Initialise PostgreSQL data directory on first run
if [ ! -f "${PG_DATA}/PG_VERSION" ]; then
    mkdir -p "${PG_DATA}"
    chown -R postgres:postgres "${PG_DATA}"

    gosu postgres "${PG_BIN}/initdb" -D "${PG_DATA}" --auth-local trust --auth-host md5

    gosu postgres "${PG_BIN}/pg_ctl" -D "${PG_DATA}" -l /tmp/pg_init.log start -w -t 30

    gosu postgres psql postgres -c "CREATE USER documind WITH SUPERUSER PASSWORD 'documind_secret';"
    gosu postgres psql postgres -c "CREATE DATABASE documind OWNER documind;"

    gosu postgres "${PG_BIN}/pg_ctl" -D "${PG_DATA}" stop -m fast -w
fi

exec /usr/bin/supervisord -n -c /etc/supervisor/supervisord.conf
