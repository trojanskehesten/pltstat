"""Tests for the dimensionality reduction helpers of ``pltstat.multfeats``.

These tests are kept apart from the other plotting tests because UMAP pulls in
numba, whose first call pays a JIT compilation cost. They are marked ``slow``
so that a quick run can skip them with ``pytest -m "not slow"``.
"""

import numpy as np
import pandas as pd
import pytest

from pltstat import multfeats as mf

from conftest import ENGINES

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def features():
    """Return a small numeric matrix with two separated clusters."""
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=0.0, scale=0.5, size=(25, 4))
    cluster_b = rng.normal(loc=5.0, scale=0.5, size=(25, 4))
    return pd.DataFrame(
        np.vstack([cluster_a, cluster_b]),
        columns=[f"f{i}" for i in range(4)],
    )


@pytest.fixture(scope="module")
def labels():
    """Return the cluster membership matching the `features` fixture."""
    return np.array([0] * 25 + [1] * 25)


@pytest.fixture(scope="module")
def embeddings(features):
    """Return the UMAP and t-SNE embeddings of the `features` fixture."""
    return mf.embeddings_creation(
        features,
        umap_kwargs={"n_neighbors": 5},
        tsne_kwargs={"perplexity": 5},
    )


class TestEmbeddingsCreation:
    """Creation of the UMAP and t-SNE embeddings."""

    def test_returns_two_embeddings_of_the_expected_shape(self, embeddings, features):
        x_umap, x_tsne = embeddings
        assert x_umap.shape == (len(features), 2)
        assert x_tsne.shape == (len(features), 2)

    def test_embeddings_are_finite(self, embeddings):
        for embedding in embeddings:
            assert np.isfinite(embedding).all()

    def test_n_components_is_honored(self, features):
        x_umap, x_tsne = mf.embeddings_creation(
            features,
            n_components=3,
            umap_kwargs={"n_neighbors": 5},
            tsne_kwargs={"perplexity": 5, "method": "exact"},
        )
        assert x_umap.shape[1] == 3
        assert x_tsne.shape[1] == 3

    def test_runs_without_standardizing(self, features):
        x_umap, x_tsne = mf.embeddings_creation(
            features,
            standardize=False,
            umap_kwargs={"n_neighbors": 5},
            tsne_kwargs={"perplexity": 5},
        )
        assert x_umap.shape[0] == len(features)
        assert x_tsne.shape[0] == len(features)


class TestPlotUmapTsne:
    """Rendering of the embeddings on both engines."""

    @pytest.mark.parametrize("engine", ENGINES)
    def test_plot_without_labels(self, embeddings, engine):
        x_umap, x_tsne = embeddings
        result = mf.plot_umap_tsne(x_umap, x_tsne, engine=engine)
        if engine == "plotly":
            import plotly.graph_objects as go

            assert isinstance(result, go.Figure)

    @pytest.mark.parametrize("engine", ENGINES)
    def test_plot_with_labels(self, embeddings, labels, engine):
        x_umap, x_tsne = embeddings
        result = mf.plot_umap_tsne(
            x_umap, x_tsne, labels=labels, title_pref="clusters", engine=engine
        )
        if engine == "plotly":
            import plotly.graph_objects as go

            assert isinstance(result, go.Figure)
