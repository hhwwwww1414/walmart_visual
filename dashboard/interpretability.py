# dashboard/interpretability.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import umap
import shap
import torch
from captum.attr import LayerActivation
import matplotlib.pyplot as plt


def plot_learning_curves(history: dict):
    """
    Построить графики обучения (loss и accuracy по эпохам).
    history = {
        "train_loss": [...],
        "val_loss":   [...],
        "train_acc":  [...],  # опционально
        "val_acc":    [...],  # опционально
    }
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))
    # Loss
    df_loss = pd.DataFrame({
        "epoch": epochs * 2,
        "loss": history["train_loss"] + history["val_loss"],
        "subset": ["train"]*len(epochs) + ["val"]*len(epochs),
    })
    fig_loss = px.line(
        df_loss, x="epoch", y="loss", color="subset", markers=True,
        title="Learning Curves — Loss"
    )
    fig_loss.update_layout(xaxis_title="Epoch", yaxis_title="Loss")

    # Accuracy, если есть
    fig_acc = None
    if "train_acc" in history and "val_acc" in history:
        df_acc = pd.DataFrame({
            "epoch": epochs * 2,
            "accuracy": history["train_acc"] + history["val_acc"],
            "subset": ["train"]*len(epochs) + ["val"]*len(epochs),
        })
        fig_acc = px.line(
            df_acc, x="epoch", y="accuracy", color="subset", markers=True,
            title="Learning Curves — Accuracy"
        )
        fig_acc.update_layout(xaxis_title="Epoch", yaxis_title="Accuracy")

    return fig_loss, fig_acc


def get_confusion_matrix_figure(y_true, y_pred, labels=[0, 1]):
    """
    Вернуть plotly-heatmap для confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(
        cm,
        text_auto="d",
        x=labels, y=labels,
        labels={"x": "Предсказано", "y": "Истинно"},
        color_continuous_scale="Blues",
        title="Confusion Matrix"
    )
    return fig


def get_roc_pr_figures(y_true, y_scores):
    """
    Вернуть два фигура: ROC curve и PR curve.
    y_scores — вероятности положительного класса.
    """
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fig_roc = px.area(
        x=fpr, y=tpr,
        title=f"ROC Curve (AUC={roc_auc:.3f})",
        labels={"x": "False Positive Rate", "y": "True Positive Rate"}
    )
    fig_roc.add_shape(
        type="line", line_dash="dash",
        x0=0, x1=1, y0=0, y1=1, line_color="gray"
    )

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(rec, prec)
    fig_pr = px.area(
        x=rec, y=prec,
        title=f"Precision-Recall Curve (AUC={pr_auc:.3f})",
        labels={"x": "Recall", "y": "Precision"}
    )

    return fig_roc, fig_pr


def get_embedding_figure(embeddings: np.ndarray, labels: np.ndarray):
    """
    Проекция эмбеддингов в 2D через UMAP.
    embeddings — np.array, shape=(n_samples, dim)
    labels — метки для раскраски точек
    """
    reducer = umap.UMAP(n_components=2, random_state=42)
    proj = reducer.fit_transform(embeddings)
    df_emb = pd.DataFrame(proj, columns=["x", "y"])
    df_emb["label"] = labels
    fig = px.scatter(
        df_emb, x="x", y="y", color="label",
        title="UMAP projection of embeddings", opacity=0.7
    )
    return fig


def compute_shap(df_sample: pd.DataFrame, model, background_size=100):
    """
    Вычислить SHAP values для модели.
    df_sample — DataFrame с роверами-признаками, размер >= background_size.
    model — обученная модель, принимающая numpy-поле размера (batch, features).
    """
    background = df_sample.iloc[:background_size].values
    explainer = shap.DeepExplainer(model, background)
    shap_vals = explainer.shap_values(df_sample.values)
    return shap_vals


def get_shap_bar_figure(shap_vals, feature_names):
    """
    Построить bar-plot по SHAP values для положительного класса shap_vals[1].
    Возвращает matplotlib.Figure.
    """
    # shap_vals: list of arrays per класс
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.summary_plot(
        shap_vals[1],
        features=None,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        ax=ax
    )
    fig.tight_layout()
    return fig


def get_feature_maps_figure(input_tensor: torch.Tensor, model: torch.nn.Module,
                            layer_name="conv1", n_maps=16):
    """
    Визуализировать n_maps первых feature maps из указанного слоя.
    input_tensor: torch.Tensor shape [1, C, H, W]
    layer_name: имя атрибута модуля в model, e.g. "conv1"
    """
    model.eval()
    # получаем модуль
    layer = getattr(model, layer_name)
    la = LayerActivation(model, layer)
    activations = la.attribute(input_tensor)  # [1, channels, H, W]
    activations = activations.detach().cpu().numpy()[0]

    cols = min(4, n_maps)
    rows = math.ceil(n_maps / cols)
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"Map {i}" for i in range(n_maps)],
                        horizontal_spacing=0.02, vertical_spacing=0.02)

    for i in range(n_maps):
        r, c = i // cols + 1, i % cols + 1
        fig.add_trace(
            go.Heatmap(z=activations[i], showscale=False),
            row=r, col=c
        )
        fig.update_xaxes(showticklabels=False, row=r, col=c)
        fig.update_yaxes(showticklabels=False, row=r, col=c)

    fig.update_layout(
        title_text=f"Feature maps from layer '{layer_name}'",
        height=200 * rows, showlegend=False, margin=dict(t=30, l=0, r=0, b=0)
    )
    return fig


def get_attention_figure(tokens: list[str], attn_weights: torch.Tensor,
                         layer=0, head=0):
    """
    Визуализировать attention-голову.
    tokens: list of токенов (строк) длины seq_len
    attn_weights: torch.Tensor shape [batch, n_heads, seq_len, seq_len]
    """
    if isinstance(attn_weights, torch.Tensor):
        arr = attn_weights.detach().cpu().numpy()
    else:
        arr = np.array(attn_weights)
    # берем batch=0, указанный слой/голову
    heat = arr[0, head]
    fig = go.Figure(data=go.Heatmap(
        z=heat,
        x=tokens, y=tokens,
        colorscale="Viridis"
    ))
    fig.update_layout(
        title=f"Attention weights (head {head})",
        xaxis_title="Query token",
        yaxis_title="Key token",
        height=600
    )
    return fig
