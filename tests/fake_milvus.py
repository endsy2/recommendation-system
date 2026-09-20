"""In-memory stand-in for pymilvus.MilvusClient, so unit tests need neither Docker nor a server.

It mirrors the method signatures and result shapes used by src.milvus_store (describe_collection
dicts, describe_index dicts, Hit-like search results keyed by the primary-key field name).
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np


class FakeIterator:
    def __init__(self, rows: list[dict[str, Any]], batch_size: int) -> None:
        self.rows, self.batch_size, self.closed = rows, batch_size, False

    def next(self) -> list[dict[str, Any]]:
        page, self.rows = self.rows[: self.batch_size], self.rows[self.batch_size:]
        return page

    def close(self) -> None:
        self.closed = True


class FakeMilvusClient:
    def __init__(self, uri: str = "http://localhost:19530", token: str = "", timeout: float | None = None, **_: Any):
        self.uri, self.token, self.timeout = uri, token, timeout
        self.collections: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.fail: Exception | None = None
        self.closed = False

    # -- helpers -----------------------------------------------------------------------------
    def _check(self, method: str, name: str = "", **kwargs: Any) -> None:
        self.calls.append((method, name, kwargs))
        if self.closed:
            raise RuntimeError("client is closed")
        if self.fail is not None:
            raise self.fail

    def _coll(self, name: str) -> dict[str, Any]:
        if name not in self.collections:
            raise RuntimeError(f"collection not found: {name}")
        return self.collections[name]

    def add_raw_collection(self, name: str, description: dict[str, Any], indexes: list[dict[str, Any]],
                           loaded: bool = True) -> None:
        pk = next(field["name"] for field in description["fields"] if field.get("is_primary"))
        self.collections[name] = {"description": description, "indexes": {i["index_name"]: i for i in indexes},
                                  "rows": {}, "pk": pk, "loaded": loaded}

    # -- MilvusClient API subset ---------------------------------------------------------------
    @staticmethod
    def prepare_index_params() -> Any:
        from pymilvus import MilvusClient

        return MilvusClient.prepare_index_params()

    def has_collection(self, name: str, timeout: float | None = None, **kwargs: Any) -> bool:
        self._check("has_collection", name)
        return name in self.collections

    def create_collection(self, name: str, schema: Any = None, index_params: Any = None, timeout: float | None = None,
                          **kwargs: Any) -> None:
        self._check("create_collection", name, **kwargs)
        if name in self.collections:
            raise RuntimeError("collection already exists")
        description = schema.to_dict()
        description.setdefault("functions", [])
        indexes = [param.to_dict() for param in index_params] if index_params is not None else []
        # Milvus auto-loads a collection created with index params.
        self.add_raw_collection(name, description, indexes, loaded=index_params is not None)

    def describe_collection(self, name: str, timeout: float | None = None, **kwargs: Any) -> dict[str, Any]:
        self._check("describe_collection", name)
        return copy.deepcopy(self._coll(name)["description"])

    def list_indexes(self, name: str, field_name: str = "", **kwargs: Any) -> list[str]:
        self._check("list_indexes", name)
        return [key for key, value in self._coll(name)["indexes"].items() if not field_name or value["field_name"] == field_name]

    def describe_index(self, name: str, index_name: str, timeout: float | None = None, **kwargs: Any) -> dict[str, Any]:
        self._check("describe_index", name)
        return dict(self._coll(name)["indexes"][index_name])

    def get_load_state(self, name: str, partition_name: str = "", timeout: float | None = None, **kwargs: Any) -> dict:
        self._check("get_load_state", name)
        if name not in self.collections:
            return {"state": "NotExist"}
        return {"state": "Loaded" if self.collections[name]["loaded"] else "NotLoad"}

    def load_collection(self, name: str, timeout: float | None = None, **kwargs: Any) -> None:
        self._check("load_collection", name)
        self._coll(name)["loaded"] = True

    def _fields(self, name: str) -> dict[str, dict[str, Any]]:
        return {field["name"]: field for field in self._coll(name)["description"]["fields"]}

    def upsert(self, name: str, data: list[dict[str, Any]], timeout: float | None = None, **kwargs: Any) -> dict:
        self._check("upsert", name, rows=len(data))
        coll = self._coll(name)
        if not coll["loaded"]:
            raise RuntimeError("collection not loaded")
        fields = self._fields(name)
        staged = {}
        for row in data:
            if set(row) != set(fields):
                raise ValueError(f"full record required: got {sorted(row)}, schema {sorted(fields)}")
            for key, value in row.items():
                field, type_name = fields[key], str(getattr(fields[key]["type"], "name", fields[key]["type"]))
                if type_name == "VARCHAR":
                    if value is None and not field.get("nullable"):
                        raise ValueError(f"{key} is not nullable")
                    if value is not None and len(value.encode("utf-8")) > field["params"]["max_length"]:
                        raise ValueError(f"{key} exceeds max_length")
                elif type_name == "FLOAT_VECTOR" and len(value) != field["params"]["dim"]:
                    raise ValueError(f"{key} has wrong dimension")
            stored = dict(row)
            for key, field in fields.items():
                if str(getattr(field["type"], "name", field["type"])) == "FLOAT_VECTOR":
                    stored[key] = [float(np.float32(x)) for x in row[key]]
            staged[row[coll["pk"]]] = stored
        coll["rows"].update(staged)
        return {"upsert_count": len(data)}

    def _project(self, name: str, row: dict[str, Any], output_fields: list[str] | None) -> dict[str, Any]:
        pk = self._coll(name)["pk"]
        keys = list(self._fields(name)) if not output_fields or output_fields == ["*"] else [pk, *output_fields]
        return {key: copy.deepcopy(row[key]) for key in dict.fromkeys(keys)}

    def query(self, name: str, filter: str = "", output_fields: list[str] | None = None, timeout: float | None = None,
              **kwargs: Any) -> list[dict[str, Any]]:
        self._check("query", name, **kwargs)
        coll = self._coll(name)
        if output_fields == ["count(*)"]:
            return [{"count(*)": len(coll["rows"])}]
        if filter:
            raise NotImplementedError("fake query supports only empty filters")
        return [self._project(name, row, output_fields) for _, row in sorted(coll["rows"].items())]

    def get(self, name: str, ids: list[Any], output_fields: list[str] | None = None, timeout: float | None = None,
            **kwargs: Any) -> list[dict[str, Any]]:
        self._check("get", name, **kwargs)
        rows = self._coll(name)["rows"]
        return [self._project(name, rows[song_id], output_fields) for song_id in ids if song_id in rows]

    def query_iterator(self, name: str, batch_size: int = 1000, filter: str = "", output_fields: list[str] | None = None,
                       timeout: float | None = None, **kwargs: Any) -> FakeIterator:
        self._check("query_iterator", name, **kwargs)
        rows = self._coll(name)["rows"]
        return FakeIterator([self._project(name, row, output_fields) for _, row in sorted(rows.items())], batch_size)

    def search(self, name: str, data: list[list[float]], anns_field: str = "", limit: int = 10,
               search_params: dict | None = None, output_fields: list[str] | None = None, timeout: float | None = None,
               **kwargs: Any) -> list[list[dict[str, Any]]]:
        self._check("search", name, anns_field=anns_field, limit=limit, search_params=search_params,
                    output_fields=output_fields, **kwargs)
        coll = self._coll(name)
        if not coll["loaded"]:
            raise RuntimeError("collection not loaded")
        results = []
        for query in data:
            q = np.asarray(query, dtype=np.float64)
            scored = []
            for song_id, row in coll["rows"].items():
                v = np.asarray(row[anns_field], dtype=np.float64)
                scored.append((float(v @ q / (np.linalg.norm(v) * np.linalg.norm(q))), song_id, row))
            scored.sort(key=lambda item: -item[0])
            results.append([{coll["pk"]: song_id, "distance": score,
                             "entity": {key: row[key] for key in (output_fields or [])}}
                            for score, song_id, row in scored[:limit]])
        return results

    def close(self) -> None:
        self.calls.append(("close", "", {}))
        self.closed = True
