# Copyright 2023 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import difflib
import functools
import logging
import pathlib
import shutil
import tarfile
import tempfile

import numpy as np
import tensorflow as tf

from typing import Optional, Dict, Any, Type
import weakref

import kubric as kb
from kubric import core
from kubric import file_io
from kubric.kubric_typing import PathLike


class LocalAssetSource(kb.AssetSource):
    """An AssetSource that loads assets from a local directory."""

    @classmethod
    def from_manifest(
        cls, manifest_path: PathLike, scratch_dir: Optional[PathLike] = None
    ) -> "LocalAssetSource":
        if manifest_path == "gs://kubric-public/assets/ShapeNetCore.v2.json":
            raise ValueError(
                f"The path `{manifest_path}` is a placeholder for the real path. "
                "Please visit https://shapenet.org, agree to terms and conditions."
                "After logging in, you will find the manifest URL here:"
                "https://shapenet.org/download/kubric"
            )

        return super().from_manifest(manifest_path, scratch_dir)

    def _resolve_asset_path(
        self, path: Optional[str], asset_id: str
    ) -> Optional[PathLike]:
        if path is None:
            return None

        return self.data_dir / path

    @staticmethod
    def _adjust_paths(
        asset_kwargs: Dict[str, Any], asset_dir: PathLike
    ) -> Dict[str, Any]:
        """If present, replace '{asset_dir}' prefix with actual asset_dir in each kwarg value."""

        def _adjust_path(p):
            if isinstance(p, str) and p.startswith("{asset_dir}/"):
                return str(asset_dir / p[12:])
            elif isinstance(p, dict):
                return {key: _adjust_path(value) for key, value in p.items()}
            else:
                return p

        return {k: _adjust_path(v) for k, v in asset_kwargs.items()}

    def create(
        self, asset_id: str, add_metadata: bool = True, **kwargs
    ) -> Type[core.Asset]:
        """
        Create an instance of an asset by a given id.

        Performs the following steps
        1. check if asset_id is found in manifest and retrieve entry
        2. determine Asset class and full path (can be remote or local cache or missing)
        3. if path is not none, then fetch and unpack the zipped asset to scratch_dir
        4. construct kwargs from asset_entry->kwargs, override with **kwargs and then
        adjust paths (ones that start with “{{asset_dir}}”
        5. create asset by calling constructor with kwargs
        6. set metadata (if add_metadata is True)
        7. return asset

        Args:
            asset_id (str): the id of the asset to be created
                            (corresponds to its key in the manifest file and
                            typically also to the filename)
            add_metadata (bool): whether to add the metadata from the asset to the instance
            **kwargs: additional kwargs to be passed to the asset constructor

        Returns:
          An instance of the specified asset (subtype of kubric.core.Asset)
        """
        # find corresponding asset entry
        asset_entry = self._assets.get(asset_id)
        if not asset_entry:
            close_matches = difflib.get_close_matches(
                asset_id, possibilities=self.all_asset_ids, n=1
            )
            if close_matches:
                raise KeyError(
                    f"Unknown asset with id='{asset_id}'. Did you mean '{close_matches[0]}'?"
                )

        # determine type and path
        asset_type = self._resolve_asset_type(asset_entry["asset_type"])
        asset_path = self._resolve_asset_path(asset_entry.get("path", ""), asset_id)

        # fetch and unpack tar.gz file if necessary
        asset_dir = None if asset_path is None else self.fetch(asset_path, asset_id)

        # construct kwargs
        asset_kwargs = asset_entry.get("kwargs", {})
        asset_kwargs.update(kwargs)
        asset_kwargs = self._adjust_paths(asset_kwargs, asset_dir)
        if asset_type == core.FileBasedObject:
            asset_kwargs["asset_id"] = asset_id
        # create the asset
        asset = asset_type(**asset_kwargs)
        # set the metadata
        if add_metadata:
            asset.metadata.update(asset_entry.get("metadata", {}))

        return asset

    def fetch(self, asset_path, asset_id):
        return asset_path / asset_id

    def get_test_split(self, fraction=0.1):
        """
        Generates a train/test split for the asset source.

        Args:
          fraction: the fraction of the asset source to use for the held-out set.

        Returns:
          train_ids: list of asset ID strings
          test_ids: list of asset ID strings
        """
        rng = np.random.default_rng(42)
        test_size = int(round(len(self.all_asset_ids) * fraction))
        test_ids = rng.choice(self.all_asset_ids, size=test_size, replace=False)
        train_ids = [i for i in self.all_asset_ids if i not in test_ids]
        return train_ids, test_ids
