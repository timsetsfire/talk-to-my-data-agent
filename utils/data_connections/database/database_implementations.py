# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import functools
import json
import logging
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, cast

import pandas as pd
import polars as pl
import snowflake.connector
from google.cloud import bigquery
from hdbcli import dbapi
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from pydantic import ValidationError

from utils.analyst_db import AnalystDB, InternalDataSourceType
from utils.code_execution import InvalidGeneratedCode
from utils.credentials import (
    DatabricksCredentials,
    GoogleCredentialsBQ,
    NoDatabaseCredentials,
    SAPDatasphereCredentials,
    SnowflakeCredentials,
)
from utils.data_connections.database.database_interface import (
    _DEFAULT_DB_QUERY_TIMEOUT,
    BigQueryCredentialArgs,
    DatabaseOperator,
    DatabricksCredentialArgs,
    NoDatabaseOperator,
    SAPDatasphereCredentialArgs,
    SnowflakeCredentialArgs,
)
from utils.data_connections.datarobot.datarobot_dataset_handler import (
    BaseRecipe,
)
from utils.prompts import (
    SYSTEM_PROMPT_BIGQUERY,
    SYSTEM_PROMPT_DATABRICKS,
    SYSTEM_PROMPT_SAP_DATASPHERE,
    SYSTEM_PROMPT_SNOWFLAKE,
)
from utils.schema import AnalystDataset, AppInfra

logger = logging.getLogger(__name__)


class SnowflakeOperator(DatabaseOperator[SnowflakeCredentialArgs]):
    def __init__(
        self,
        credentials: SnowflakeCredentials,
        default_timeout: int = _DEFAULT_DB_QUERY_TIMEOUT,
    ):
        if not credentials.is_configured():
            raise ValueError("Snowflake credentials not properly configured")
        self._credentials = credentials
        self.default_timeout = default_timeout

    @asynccontextmanager
    async def create_connection(
        self,
    ) -> AsyncGenerator[snowflake.connector.SnowflakeConnection, None]:
        """Create a connection to Snowflake using environment variables"""
        if not self._credentials.is_configured():
            raise ValueError("Snowflake credentials not properly configured")

        connect_params: dict[str, Any] = {
            "user": self._credentials.user,
            "account": self._credentials.account,
            "warehouse": self._credentials.warehouse,
            "database": self._credentials.database,
            "schema": self._credentials.db_schema,
            "role": self._credentials.role,
        }

        # Try key file authentication first if configured
        if private_key := self._credentials.get_private_key():
            connect_params["private_key"] = private_key
        elif self._credentials.password:
            connect_params["password"] = self._credentials.password
        else:
            raise ValueError(
                "Neither private key nor password authentication configured"
            )

        # In some enviroments, the Snowflake client's platform detection crashes. This patch skips that detection.
        snowflake.connector.SnowflakeConnection.platform_detection_timeout_seconds = 0.0  # type: ignore[method-assign,assignment]

        loop = asyncio.get_running_loop()
        connection = await loop.run_in_executor(
            None, lambda: snowflake.connector.connect(**connect_params)
        )
        try:
            yield connection
        finally:
            connection.close()

    async def execute_query(
        self, query: str, timeout: int | None = None
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        """Execute a Snowflake query with timeout and metadata capture

        Args:
            conn: Snowflake connection
            query: SQL query to execute
            timeout: Query timeout in seconds

        Returns:
            Tuple of (results, metadata)
        """
        timeout = timeout if timeout is not None else self.default_timeout

        loop = asyncio.get_running_loop()

        try:
            async with self.create_connection() as conn:
                with await loop.run_in_executor(
                    None, conn.cursor, snowflake.connector.DictCursor
                ) as cursor:
                    # Set query timeout at cursor level
                    await loop.run_in_executor(
                        None,
                        cursor.execute,
                        f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {timeout}",
                    )

                    try:
                        # Execute query
                        await loop.run_in_executor(None, cursor.execute, query)

                        # Get results
                        results = await loop.run_in_executor(None, cursor.fetchall)

                        return results

                    except snowflake.connector.errors.ProgrammingError as e:
                        # Handle Snowflake-specific errors
                        raise InvalidGeneratedCode(
                            f"Snowflake error: {str(e.msg)}",
                            code=query,
                            exception=None,
                            traceback_str="",
                        )

        except Exception as e:
            raise InvalidGeneratedCode(
                f"Query execution failed: {str(e)}",
                code=query,
                exception=e,
                traceback_str=traceback.format_exc(),
            )

    async def get_tables(self, timeout: int | None = None) -> list[str]:
        """Fetch list of tables from Snowflake schema"""
        timeout = timeout if timeout is not None else self.default_timeout
        loop = asyncio.get_running_loop()

        conn: snowflake.connector.SnowflakeConnection
        try:
            async with self.create_connection() as conn:
                with await loop.run_in_executor(None, conn.cursor) as cursor:
                    # Log current session info
                    logger.info("Checking current session settings...")
                    await loop.run_in_executor(
                        None,
                        cursor.execute,
                        f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {timeout}",
                    )

                    await loop.run_in_executor(
                        None,
                        cursor.execute,
                        "SELECT CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_ROLE(), CURRENT_WAREHOUSE()",
                    )
                    current_settings = await loop.run_in_executor(None, cursor.fetchone)
                    logger.info("Current settings %s.", current_settings)
                    logger.info(
                        f"Current settings - Database: {current_settings[0]}, Schema: {current_settings[1]}, Role: {current_settings[2]}, Warehouse: {current_settings[3]}"  # type: ignore[index]
                    )

                    # Check if schema exists
                    await loop.run_in_executor(
                        None,
                        cursor.execute,
                        f"""
                        SELECT COUNT(*)
                        FROM {self._credentials.database}.INFORMATION_SCHEMA.SCHEMATA
                        WHERE SCHEMA_NAME = '{self._credentials.db_schema}'
                    """,
                    )
                    schema_exists = (await loop.run_in_executor(None, cursor.fetchone))[  # type: ignore[index]
                        0
                    ]
                    logger.info(f"Schema exists check: {schema_exists > 0}")

                    # Get all objects (tables and views)
                    await loop.run_in_executor(
                        None,
                        cursor.execute,
                        f"""
                        SELECT table_name, table_type
                        FROM {self._credentials.database}.information_schema.tables
                        WHERE table_schema = '{self._credentials.db_schema}'
                        AND table_type IN ('BASE TABLE', 'VIEW')
                        ORDER BY table_type, table_name
                    """,
                    )
                    results = await loop.run_in_executor(None, cursor.fetchall)
                    tables = [row[0] for row in results]

                    # Log detailed results
                    logger.info(f"Total objects found: {len(results)}")
                    for table_name, table_type in results:
                        logger.info(f"Found {table_type}: {table_name}")

                    # Check schema privileges
                    await loop.run_in_executor(
                        None,
                        cursor.execute,
                        f"""
                        SHOW GRANTS ON SCHEMA {self._credentials.database}.{self._credentials.db_schema}
                    """,
                    )
                    privileges = await loop.run_in_executor(None, cursor.fetchall)
                    logger.info("Schema privileges:")
                    for priv in privileges:
                        logger.info(f"Privilege: {priv}")

                    return tables

        except Exception:
            logger.error("Failed to fetch tables.", exc_info=True)
            return []

    @functools.lru_cache(maxsize=8)
    async def get_data(
        self,
        *table_names: str,
        analyst_db: AnalystDB,
        sample_size: int = 5000,
        timeout: int | None = None,
    ) -> list[str]:
        """Load selected tables from Snowflake as pandas DataFrames

        Args:
        - table_names: List of table names to fetch
        - sample_size: Number of rows to sample from each table

        Returns:
        - Dictionary of table names to list of records
        """

        timeout = timeout if timeout is not None else self.default_timeout
        loop = asyncio.get_running_loop()

        conn: snowflake.connector.SnowflakeConnection

        dataframes = []
        try:
            async with self.create_connection() as conn:
                with await loop.run_in_executor(None, conn.cursor) as cursor:
                    for table in table_names:
                        try:
                            qualified_table = f'{self._credentials.database}.{self._credentials.db_schema}."{table}"'
                            logger.info(f"Fetching data from table: {qualified_table}")
                            await loop.run_in_executor(
                                None,
                                cursor.execute,
                                f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {timeout}",
                            )
                            await loop.run_in_executor(
                                None,
                                cursor.execute,
                                f"DESCRIBE TABLE {qualified_table}",
                            )
                            column_response = await loop.run_in_executor(
                                None, cursor.fetchall
                            )

                            column_types = {row[0]: row[1] for row in column_response}

                            await loop.run_in_executor(
                                None,
                                cursor.execute,
                                f"""
                                SELECT * FROM {qualified_table}
                                SAMPLE ({sample_size} ROWS)
                            """,
                            )

                            columns = [desc[0] for desc in cursor.description]
                            data = await loop.run_in_executor(None, cursor.fetchall)
                            schema = [
                                {
                                    "name": col,
                                    "dataType": column_types.get(col, "VARCHAR"),
                                }
                                for col in columns
                            ]

                            df = BaseRecipe.convert_preview_to_dataframe(schema, data)

                            logger.info(
                                f"Successfully loaded table {table}: {len(df.df)} rows, {len(df.df.columns)} columns"
                            )
                            dataframes.append(
                                (AnalystDataset(name=table, data=df), column_types)
                            )

                        except Exception as e:
                            logger.error(
                                f"Error loading table {table}: {str(e)}", exc_info=True
                            )
                            continue
                names = []
                for dataframe, column_types in dataframes:
                    await analyst_db.register_dataset(
                        dataframe,
                        InternalDataSourceType.DATABASE,
                        original_column_types=column_types,
                        clobber=True,
                    )
                    names.append(dataframe.name)
                return names

        except Exception as e:
            logger.error(f"Error fetching Snowflake data: {str(e)}", exc_info=True)
            return []

    def get_system_prompt(self) -> ChatCompletionSystemMessageParam:
        return ChatCompletionSystemMessageParam(
            role="system",
            content=SYSTEM_PROMPT_SNOWFLAKE.format(
                warehouse=self._credentials.warehouse,
                database=self._credentials.database,
                schema=self._credentials.db_schema,
            ),
        )


class BigQueryOperator(DatabaseOperator[BigQueryCredentialArgs]):
    def __init__(
        self,
        credentials: GoogleCredentialsBQ,
        default_timeout: int = _DEFAULT_DB_QUERY_TIMEOUT,
    ):
        self._credentials = credentials
        self._credentials.db_schema = self._credentials.db_schema
        self._database = credentials.service_account_key["project_id"]
        self.default_timeout = default_timeout

    @asynccontextmanager
    async def create_connection(self) -> AsyncGenerator[bigquery.Client, None]:
        from google.oauth2 import service_account

        loop = asyncio.get_running_loop()
        google_credentials = await loop.run_in_executor(
            None,
            lambda: service_account.Credentials.from_service_account_info(  # type: ignore[no-untyped-call]
                self._credentials.service_account_key,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            ),
        )
        with bigquery.Client(
            credentials=google_credentials,
        ) as client:
            yield client

    async def execute_query(
        self, query: str, timeout: int | None = None
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        timeout = timeout if timeout is not None else self.default_timeout
        loop = asyncio.get_running_loop()
        try:
            async with self.create_connection() as conn:
                results = await loop.run_in_executor(
                    None, lambda: conn.query(query, timeout=timeout)
                )

                sql_result: pd.DataFrame = await loop.run_in_executor(
                    None, results.to_dataframe
                )

                sql_result_as_dicts = cast(
                    list[dict[str, Any]], sql_result.to_dict(orient="records")
                )
                return sql_result_as_dicts

        except Exception as e:
            raise InvalidGeneratedCode(
                f"Query execution failed: {str(e)}",
                code=query,
                exception=e,
                traceback_str=traceback.format_exc(),
            )

    async def get_tables(self, timeout: int | None = None) -> list[str]:
        """Fetch list of tables from BigQuery schema"""
        timeout = timeout if timeout is not None else self.default_timeout
        loop = asyncio.get_running_loop()

        try:
            async with self.create_connection() as conn:
                tables = [
                    i.table_id
                    for i in await loop.run_in_executor(
                        None,
                        lambda: conn.list_tables(
                            str(self._credentials.db_schema), timeout=timeout
                        ),
                    )
                ]

                # Log detailed results
                logger.info(f"Total objects found: {len(tables)}")
                logger.info(f"Found tables: {', '.join(tables)}")

                return tables

        except Exception as e:
            logger.error(f"Failed to fetch tables: {str(e)}", exc_info=True)
            return []

    @functools.lru_cache(maxsize=8)
    async def get_data(
        self,
        *table_names: str,
        analyst_db: AnalystDB,
        sample_size: int = 5000,
        timeout: int | None = None,
    ) -> list[str]:
        timeout = timeout if timeout is not None else self.default_timeout
        loop = asyncio.get_running_loop()

        dataframes = []

        try:
            async with self.create_connection() as conn:
                for table in table_names:
                    try:
                        qualified_table = (
                            f"{self._database}.{self._credentials.db_schema}.{table}"
                        )
                        logger.info(f"Fetching data from table: {qualified_table}")
                        column_type_query = await loop.run_in_executor(
                            None,
                            conn.query,
                            f"""
                            SELECT column_name, data_type FROM `{self._database}.{self._credentials.db_schema}.INFORMATION_SCHEMA.COLUMNS` 
                            WHERE table_name = '{table}' ORDER BY ordinal_position
                        """,
                        )
                        column_type_results = await loop.run_in_executor(
                            None, column_type_query.result
                        )

                        column_types = {row[0]: row[1] for row in column_type_results}

                        results = await loop.run_in_executor(
                            None,
                            lambda: conn.query(
                                f"""
                                SELECT * FROM `{qualified_table}`
                                LIMIT {sample_size}
                            """,
                                timeout=timeout,
                            ),
                        )

                        pandas_df: pd.DataFrame = await loop.run_in_executor(
                            None, results.to_dataframe
                        )
                        df = pl.from_pandas(pandas_df)
                        logger.info(
                            f"Successfully loaded table {table}: {len(df)} rows, {len(df.columns)} columns"
                        )

                        dataframes.append(
                            (AnalystDataset(name=table, data=df), column_types)
                        )

                    except Exception as e:
                        logger.error(
                            f"Error loading table {table}: {str(e)}", exc_info=True
                        )
                        continue

                names = []
                for dataframe, column_types in dataframes:
                    await analyst_db.register_dataset(
                        dataframe,
                        InternalDataSourceType.DATABASE,
                        clobber=True,
                        original_column_types=column_types,
                    )
                    names.append(dataframe.name)

                return names

        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}", exc_info=True)
            return []

    def get_system_prompt(self) -> ChatCompletionSystemMessageParam:
        return ChatCompletionSystemMessageParam(
            role="system",
            content=SYSTEM_PROMPT_BIGQUERY.format(
                project=self._database,
                dataset=self._credentials.db_schema,
            ),
        )


class SAPDatasphereOperator(DatabaseOperator[SAPDatasphereCredentialArgs]):
    def __init__(
        self,
        credentials: SAPDatasphereCredentials,
        default_timeout: int = _DEFAULT_DB_QUERY_TIMEOUT,
    ):
        if not credentials.is_configured():
            raise ValueError("SAP Data Sphere credentials not properly configured")
        self._credentials = credentials
        self.default_timeout = default_timeout

    @asynccontextmanager
    async def create_connection(self) -> AsyncGenerator[dbapi.Connection, None]:
        """Create a connection to SAP Data Sphere"""
        if not self._credentials.is_configured():
            raise ValueError("SAP Data Sphere credentials not properly configured")

        connect_params: dict[str, Any] = {
            "address": self._credentials.host,
            "port": self._credentials.port,
            "user": self._credentials.user,
            "password": self._credentials.password,
        }

        loop = asyncio.get_running_loop()

        connection = await loop.run_in_executor(
            None, lambda: dbapi.connect(**connect_params)
        )
        try:
            yield connection
        finally:
            connection.close()

    async def execute_query(
        self, query: str, timeout: int | None = None
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        """Execute a SAP Data Sphere query with timeout

        Args:
            query: SQL query to execute
            timeout: Query timeout in seconds

        Returns:
            Query results
        """
        timeout = timeout if timeout is not None else self.default_timeout
        loop = asyncio.get_running_loop()
        try:
            async with self.create_connection() as conn:
                with await loop.run_in_executor(None, conn.cursor) as cursor:
                    try:
                        # Execute query
                        await loop.run_in_executor(None, cursor.execute, query)

                        # Get results
                        results = await loop.run_in_executor(None, cursor.fetchall)

                        return [
                            dict(zip(row.column_names, row.column_values))
                            for row in results
                        ]

                    except Exception as e:
                        # Handle SAP Data Sphere specific errors
                        raise InvalidGeneratedCode(
                            f"SAP Data Sphere error: {str(e)}",
                            code=query,
                            exception=None,
                            traceback_str="",
                        )

        except Exception as e:
            raise InvalidGeneratedCode(
                f"Query execution failed: {str(e)}",
                code=query,
                exception=e,
                traceback_str=traceback.format_exc(),
            )

    async def get_tables(self, timeout: int | None = None) -> list[str]:
        """Fetch list of tables from SAP Data Sphere schema"""
        timeout = timeout if timeout is not None else self.default_timeout
        loop = asyncio.get_running_loop()

        try:
            async with self.create_connection() as conn:
                with await loop.run_in_executor(None, conn.cursor) as cursor:
                    # Get all tables and views in the schema
                    await loop.run_in_executor(
                        None,
                        cursor.execute,
                        f"""
                        SELECT TABLE_NAME
                        FROM SYS.TABLES
                        WHERE SCHEMA_NAME = '{self._credentials.db_schema}'
                        ORDER BY TABLE_NAME
                        """,
                    )
                    tables = [
                        row[0]
                        for row in await loop.run_in_executor(None, cursor.fetchall)
                    ]

                    # Get all views
                    await loop.run_in_executor(
                        None,
                        cursor.execute,
                        f"""
                        SELECT VIEW_NAME
                        FROM SYS.VIEWS
                        WHERE SCHEMA_NAME = '{self._credentials.db_schema}'
                        ORDER BY VIEW_NAME
                        """,
                    )
                    views = [
                        row[0]
                        for row in await loop.run_in_executor(None, cursor.fetchall)
                    ]

                    all_objects = tables + views

                    # Log detailed results
                    logger.info(
                        f"Total objects found in schema {self._credentials.db_schema}: {len(all_objects)}"
                    )
                    logger.info(f"Tables: {len(tables)}, Views: {len(views)}")

                    return all_objects

        except Exception as e:
            logger.error(
                f"Failed to fetch tables from SAP Data Sphere: {str(e)}", exc_info=True
            )
            return []

    @functools.lru_cache(maxsize=8)
    async def get_data(
        self,
        *table_names: str,
        analyst_db: AnalystDB,
        sample_size: int = 5000,
        timeout: int | None = None,
    ) -> list[str]:
        """Load selected tables from SAP Data Sphere as DataFrames

        Args:
        - table_names: List of table names to fetch
        - sample_size: Number of rows to sample from each table
        - timeout: Query timeout in seconds

        Returns:
        - List of registered dataset names
        """
        timeout = timeout if timeout is not None else self.default_timeout
        loop = asyncio.get_running_loop()
        dataframes = []

        try:
            async with self.create_connection() as conn:
                with await loop.run_in_executor(None, conn.cursor) as cursor:
                    for table in table_names:
                        try:
                            qualified_table = (
                                f'"{self._credentials.db_schema}"."{table}"'
                            )
                            logger.info(f"Fetching data from table: {qualified_table}")
                            await loop.run_in_executor(
                                None,
                                cursor.execute,
                                f"SELECT COLUMN_NAME, DATA_TYPE_NAME FROM SYS.TABLE_COLUMNS WHERE TABLE_NAME = '{table}' AND SCHEMA_NAME = '{self._credentials.db_schema}'",
                            )

                            column_types = {
                                row[0]: row[1]
                                for row in await loop.run_in_executor(
                                    None, cursor.fetchall
                                )
                            }

                            # Execute query to get data with limit
                            await loop.run_in_executor(
                                None,
                                cursor.execute,
                                f"""
                                SELECT * FROM {qualified_table}
                                LIMIT {sample_size}
                                """,
                            )

                            # Get column names
                            columns = [desc[0] for desc in cursor.description]
                            schema = [
                                {
                                    "name": col,
                                    "dataType": column_types.get(col, "VARCHAR"),
                                }
                                for col in columns
                            ]

                            data = await loop.run_in_executor(None, cursor.fetchall)

                            df = BaseRecipe.convert_preview_to_dataframe(
                                schema,
                                data,
                            )

                            logger.info(
                                f"Successfully loaded table {table}: {len(df.df)} rows, {len(df.df.columns)} columns"
                            )
                            dataframes.append(
                                (AnalystDataset(name=table, data=df), column_types)
                            )

                        except Exception as e:
                            logger.error(
                                f"Error loading table {table}: {str(e)}", exc_info=True
                            )
                            continue

                # Register datasets
                names = []
                for dataframe, column_types in dataframes:
                    await analyst_db.register_dataset(
                        dataframe,
                        InternalDataSourceType.DATABASE,
                        original_column_types=column_types,
                        clobber=True,
                    )
                    names.append(dataframe.name)
                return names

        except Exception as e:
            logger.error(
                f"Error fetching SAP Data Sphere data: {str(e)}", exc_info=True
            )
            return []

    def get_system_prompt(self) -> ChatCompletionSystemMessageParam:
        return ChatCompletionSystemMessageParam(
            role="system",
            content=SYSTEM_PROMPT_SAP_DATASPHERE.format(
                schema=self._credentials.db_schema,
            ),
        )


class DatabricksOperator(DatabaseOperator[DatabricksCredentialArgs]):
    """Operator for connecting to Databricks SQL warehouses.
    
    This operator connects to Databricks using the databricks-sql-connector
    and can query Delta Lake tables through Unity Catalog or the legacy
    hive_metastore catalog.
    """

    def __init__(
        self,
        credentials: DatabricksCredentials,
        default_timeout: int = _DEFAULT_DB_QUERY_TIMEOUT,
    ):
        if not credentials.is_configured():
            raise ValueError("Databricks credentials not properly configured")
        self._credentials = credentials
        self.default_timeout = default_timeout

    @asynccontextmanager
    async def create_connection(self) -> AsyncGenerator[Any, None]:
        """Create a connection to Databricks SQL warehouse"""
        from databricks import sql as databricks_sql

        if not self._credentials.is_configured():
            raise ValueError("Databricks credentials not properly configured")

        loop = asyncio.get_running_loop()

        connection = await loop.run_in_executor(
            None,
            lambda: databricks_sql.connect(
                server_hostname=self._credentials.server_hostname,
                http_path=self._credentials.http_path,
                access_token=self._credentials.access_token,
                catalog=self._credentials.catalog,
                schema=self._credentials.db_schema,
            ),
        )
        try:
            yield connection
        finally:
            connection.close()

    async def execute_query(
        self, query: str, timeout: int | None = None
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        """Execute a Databricks SQL query with timeout

        Args:
            query: SQL query to execute
            timeout: Query timeout in seconds

        Returns:
            Query results as list of dictionaries
        """
        timeout = timeout if timeout is not None else self.default_timeout
        loop = asyncio.get_running_loop()

        try:
            async with self.create_connection() as conn:
                cursor = await loop.run_in_executor(None, conn.cursor)
                try:
                    # Execute query
                    await loop.run_in_executor(None, cursor.execute, query)

                    # Get column names from description
                    columns = [desc[0] for desc in cursor.description]

                    # Get results
                    rows = await loop.run_in_executor(None, cursor.fetchall)

                    # Convert to list of dicts
                    results = [dict(zip(columns, row)) for row in rows]

                    return results

                except Exception as e:
                    raise InvalidGeneratedCode(
                        f"Databricks SQL error: {str(e)}",
                        code=query,
                        exception=None,
                        traceback_str="",
                    )
                finally:
                    cursor.close()

        except Exception as e:
            raise InvalidGeneratedCode(
                f"Query execution failed: {str(e)}",
                code=query,
                exception=e,
                traceback_str=traceback.format_exc(),
            )

    async def get_tables(self, timeout: int | None = None) -> list[str]:
        """Fetch list of tables from Databricks schema
        
        Handles both Unity Catalog and legacy Hive metastore (spark_catalog).
        """
        timeout = timeout if timeout is not None else self.default_timeout
        loop = asyncio.get_running_loop()

        try:
            async with self.create_connection() as conn:
                cursor = await loop.run_in_executor(None, conn.cursor)
                try:
                    # Try Unity Catalog's information_schema first
                    # Fall back to SHOW TABLES for legacy hive metastore (spark_catalog)
                    try:
                        await loop.run_in_executor(
                            None,
                            cursor.execute,
                            f"""
                            SELECT table_name
                            FROM `{self._credentials.catalog}`.information_schema.tables
                            WHERE table_schema = '{self._credentials.db_schema}'
                            AND table_type IN ('MANAGED', 'EXTERNAL', 'VIEW')
                            ORDER BY table_name
                            """,
                        )
                        rows = await loop.run_in_executor(None, cursor.fetchall)
                        tables = [row[0] for row in rows]
                    except Exception as info_schema_error:
                        # Fallback for legacy hive metastore (spark_catalog)
                        logger.info(
                            f"information_schema not available, using SHOW TABLES: {info_schema_error}"
                        )
                        await loop.run_in_executor(
                            None,
                            cursor.execute,
                            f"SHOW TABLES IN `{self._credentials.catalog}`.`{self._credentials.db_schema}`",
                        )
                        rows = await loop.run_in_executor(None, cursor.fetchall)
                        # SHOW TABLES returns: (database, tableName, isTemporary)
                        tables = [row[1] if len(row) > 1 else row[0] for row in rows]

                    logger.info(f"Total objects found: {len(tables)}")
                    for table_name in tables:
                        logger.info(f"Found table: {table_name}")

                    return tables

                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Failed to fetch tables from Databricks: {str(e)}", exc_info=True)
            return []

    @functools.lru_cache(maxsize=8)
    async def get_data(
        self,
        *table_names: str,
        analyst_db: AnalystDB,
        sample_size: int = 5000,
        timeout: int | None = None,
    ) -> list[str]:
        """Load selected tables from Databricks as DataFrames

        Args:
        - table_names: List of table names to fetch
        - sample_size: Number of rows to sample from each table
        - timeout: Query timeout in seconds

        Returns:
        - List of registered dataset names
        """
        timeout = timeout if timeout is not None else self.default_timeout
        loop = asyncio.get_running_loop()
        dataframes = []

        try:
            async with self.create_connection() as conn:
                cursor = await loop.run_in_executor(None, conn.cursor)
                try:
                    for table in table_names:
                        try:
                            qualified_table = f"`{self._credentials.catalog}`.`{self._credentials.db_schema}`.`{table}`"
                            logger.info(f"Fetching data from table: {qualified_table}")

                            # Get column types - try information_schema first, fall back to DESCRIBE
                            try:
                                await loop.run_in_executor(
                                    None,
                                    cursor.execute,
                                    f"""
                                    SELECT column_name, data_type
                                    FROM `{self._credentials.catalog}`.information_schema.columns
                                    WHERE table_schema = '{self._credentials.db_schema}'
                                    AND table_name = '{table}'
                                    ORDER BY ordinal_position
                                    """,
                                )
                                column_type_rows = await loop.run_in_executor(
                                    None, cursor.fetchall
                                )
                                column_types = {row[0]: row[1] for row in column_type_rows}
                            except Exception:
                                # Fallback for legacy hive metastore - use DESCRIBE TABLE
                                await loop.run_in_executor(
                                    None,
                                    cursor.execute,
                                    f"DESCRIBE TABLE {qualified_table}",
                                )
                                describe_rows = await loop.run_in_executor(
                                    None, cursor.fetchall
                                )
                                # DESCRIBE returns: (col_name, data_type, comment)
                                column_types = {row[0]: row[1] for row in describe_rows if row[0] and not row[0].startswith("#")}

                            # Fetch data with limit
                            await loop.run_in_executor(
                                None,
                                cursor.execute,
                                f"""
                                SELECT * FROM {qualified_table}
                                LIMIT {sample_size}
                                """,
                            )

                            columns = [desc[0] for desc in cursor.description]
                            data = await loop.run_in_executor(None, cursor.fetchall)
                            schema = [
                                {
                                    "name": col,
                                    "dataType": column_types.get(col, "STRING"),
                                }
                                for col in columns
                            ]

                            df = BaseRecipe.convert_preview_to_dataframe(schema, data)

                            logger.info(
                                f"Successfully loaded table {table}: {len(df.df)} rows, {len(df.df.columns)} columns"
                            )
                            dataframes.append(
                                (AnalystDataset(name=table, data=df), column_types)
                            )

                        except Exception as e:
                            logger.error(
                                f"Error loading table {table}: {str(e)}", exc_info=True
                            )
                            continue

                finally:
                    cursor.close()

                # Register datasets
                names = []
                for dataframe, column_types in dataframes:
                    await analyst_db.register_dataset(
                        dataframe,
                        InternalDataSourceType.DATABASE,
                        original_column_types=column_types,
                        clobber=True,
                    )
                    names.append(dataframe.name)
                return names

        except Exception as e:
            logger.error(f"Error fetching Databricks data: {str(e)}", exc_info=True)
            return []

    def get_system_prompt(self) -> ChatCompletionSystemMessageParam:
        return ChatCompletionSystemMessageParam(
            role="system",
            content=SYSTEM_PROMPT_DATABRICKS.format(
                catalog=self._credentials.catalog,
                schema=self._credentials.db_schema,
            ),
        )

    def query_friendly_name(self, dataset_name: str) -> str:
        """Return a fully qualified table name for Databricks."""
        return f"`{self._credentials.catalog}`.`{self._credentials.db_schema}`.`{dataset_name}`"


def get_database_operator(app_infra: AppInfra) -> DatabaseOperator[Any]:
    logger.info("Loading database %s", app_infra.database)
    if app_infra.database == "bigquery":
        credentials: (
            GoogleCredentialsBQ
            | SnowflakeCredentials
            | SAPDatasphereCredentials
            | DatabricksCredentials
            | NoDatabaseCredentials
        )
        try:
            credentials = GoogleCredentialsBQ()
            if credentials.service_account_key and credentials.db_schema:
                return cast(DatabaseOperator[Any], BigQueryOperator(credentials))
        except (ValidationError, ValueError):
            logger.warning(
                "BigQuery credentials not properly configured, falling back to no database",
                exc_info=True,
            )
        return NoDatabaseOperator(NoDatabaseCredentials())
    elif app_infra.database == "snowflake":
        try:
            credentials = SnowflakeCredentials()
            if credentials.is_configured():
                return cast(DatabaseOperator[Any], SnowflakeOperator(credentials))
        except (ValidationError, ValueError):
            logger.warning(
                "Snowflake credentials not properly configured, falling back to no database"
            )
        return NoDatabaseOperator(NoDatabaseCredentials())
    elif app_infra.database == "sap":
        try:
            credentials = SAPDatasphereCredentials()
            if credentials.is_configured():
                return cast(DatabaseOperator[Any], SAPDatasphereOperator(credentials))
        except (ValidationError, ValueError):
            logger.warning(
                "SAP credentials not properly configured, falling back to no database"
            )
        return NoDatabaseOperator(NoDatabaseCredentials())
    elif app_infra.database == "databricks":
        try:
            credentials = DatabricksCredentials()
            if credentials.is_configured():
                return cast(DatabaseOperator[Any], DatabricksOperator(credentials))
        except (ValidationError, ValueError) as e:
            logger.warning(f"Databricks credentials not properly configured: {str(e)}", exc_info=True)
            logger.warning(
                "Databricks credentials not properly configured, falling back to no database"
            )
        return NoDatabaseOperator(NoDatabaseCredentials())
    else:
        return NoDatabaseOperator(NoDatabaseCredentials())


def load_app_infra() -> AppInfra:
    directories = [".", "frontend", "app_backend"]
    error = None
    for directory in directories:
        path = Path(directory).joinpath("app_infra.json")
        try:
            with open(path) as infra_selection:
                app_json = json.load(infra_selection)
                return AppInfra(**app_json)
        except (FileNotFoundError, ValidationError) as e:
            error = e
    raise ValueError(
        "Failed to read app_infra.json.\n"
        "If running locally, verify you have selected the correct "
        "stack and that it is active using `pulumi stack output`.\n"
        f"Ensure file is created by running `pulumi up`: {str(error)}"
    ) from error


def get_external_database() -> DatabaseOperator[Any]:
    return get_database_operator(load_app_infra())
