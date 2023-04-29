import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_flops(
        data_dict,
        save_path,
        fig_size=(12, 6),
        bar_width=0.25,
        font_label=24,
        label_pad=12,
        x_tick_label_size=22,
        y_tick_label_size=22,
        x_label_rotation=30,
        y_label_rotation=0,
        plt_title="",
        y_ticks=[],
        title_font_size=29,
        show_legend=False,
        legend_font_size=14,
        legend_title_font_size=14,
        legend_loc="upper right",
        legend_bbox_to_anchor=(1.3, 0.95),
        fig_name="",
        dpi=500,
        show_title=True,
        category_names=None,
):
    data_df = pd.DataFrame(data_dict)

    plt.style.use("ggplot")
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    label = data_df["Dataset"]
    x = np.arange(len(label))

    rect1 = ax.bar(
        x - 2 * bar_width,
        data_df["MoIE_no_finetuned"],
        width=bar_width,
        label=data_df["cov_no_finetuned"],
        edgecolor="black",
        color="tab:red",
    )

    for rect, bar_label in zip(rect1, data_df["cov_no_finetuned"]):
        height = rect.get_height() * 1.2
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            height,
            bar_label,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=14,
        )

    rect2 = ax.bar(
        x - bar_width,
        data_df["MoIE_finetuned"],
        width=bar_width,
        label=data_df["cov_finetuned"],
        edgecolor="black",
        color="tab:blue",
    )
    for rect, bar_label in zip(rect2, data_df["cov_finetuned"]):
        height = rect.get_height() * 1.2
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            height,
            bar_label,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=14,
        )

    rect3 = ax.bar(
        x,
        data_df["MoIE_r_no_finetuned"],
        width=bar_width,
        label="MoIE_r_no_finetuned",
        edgecolor="black",
        color="orange",
    )
    rect4 = ax.bar(
        x + bar_width,
        data_df["MoIE_r_finetuned"],
        width=bar_width,
        label="MoIE_r_finetuned",
        edgecolor="black",
        color="green",
    )
    rect5 = ax.bar(
        x + 2 * bar_width,
        data_df["BB"],
        width=bar_width,
        label="MoIE_r_finetuned",
        edgecolor="black",
        color="gray",
    )

    plt.yscale("log")
    ax.set_ylabel("log(Flops (T))", fontsize=font_label, labelpad=label_pad)
    ax.set_xlabel("% of training samples", fontsize=font_label, labelpad=label_pad)

    handles, _ = ax.get_legend_handles_labels()

    ax.set_xticks(x)
    ax.set_xticklabels(label)
    #     ax.set_yticks(y_ticks)
    ax.tick_params(
        axis="x", labelrotation=x_label_rotation, labelsize=x_tick_label_size
    )
    ax.tick_params(
        axis="y",
        labelrotation=y_label_rotation,
        labelsize=y_tick_label_size,
    )

    if show_legend:
        myl = [rect1] + [rect2] + [rect3] + [rect4] + [rect5]
        category_names = category_names
        plt.legend(
            category_names,
            loc="upper left",
            mode="expand",
            ncol=5,
            fontsize=legend_font_size,
        )
    if show_title:
        plt.title(
            plt_title,
            fontweight="bold",
            fontsize=title_font_size,
        )
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{fig_name}"), dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_residual_vs_BB(
        data_dict, save_path, bb_threshold, fig_size=(12, 6), bar_width=0.25, font_label=24, label_pad=12,
        x_tick_label_size=22, y_tick_label_size=22, x_label_rotation=30, y_label_rotation=0, plt_title="", y_ticks=[],
        title_font_size=29, show_legend=False, legend_font_size=14, legend_title_font_size=14, legend_loc="upper right",
        legend_bbox_to_anchor=(1.3, 0.95), fig_name="", dpi=500, show_title=True,
        category_names=None
):
    data_df = pd.DataFrame(data_dict)

    plt.style.use("ggplot")
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    label = data_df["Dataset"]
    x = np.arange(len(label))

    rect1 = ax.bar(x - bar_width, data_df["BB"], width=bar_width, label="BB", edgecolor="black")
    rect2 = ax.bar(x, data_df["Residual"], width=bar_width, label="G", edgecolor="black")

    ax.set_ylabel("AUROC", fontsize=font_label, labelpad=label_pad)
    ax.set_xlabel("Iterations", fontsize=font_label, labelpad=label_pad)

    l7 = plt.axhline(y=bb_threshold, linewidth=4, color="#770060", linestyle="--", label="Blackbox")

    handles, _ = ax.get_legend_handles_labels()

    ax.set_xticks(x)
    ax.set_xticklabels(label)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis="x", labelrotation=x_label_rotation, labelsize=x_tick_label_size)
    ax.tick_params(axis="y", labelrotation=y_label_rotation, labelsize=y_tick_label_size, )

    if show_legend:
        myl = [rect1] + [rect2] + [l7]
        category_names = category_names
        lgd = ax.legend(
            myl, category_names, fontsize=legend_font_size, title_fontsize=legend_title_font_size, loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor
        )
    if show_title:
        plt.title(plt_title, fontweight="bold", fontsize=title_font_size, )
    plt.savefig(os.path.join(save_path, f"{fig_name}"), dpi=dpi, bbox_inches="tight")
    plt.show()


def cum_plot_coverages(
        results, category_colors, category_names, save_path, fig_size=(11, 2.5), plt_title_font=29, x_tick_font=22,
        title="", fig_name="", dpi=500
):
    plt.style.use("ggplot")
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = category_colors

    fig, ax = plt.subplots(figsize=fig_size)
    plt.tight_layout()
    ax.set_xlim(0, np.sum(data, axis=1).max())
    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=color)

    plt.title(title, fontweight="bold", fontsize=plt_title_font)
    ax.set_yticks([])
    ax.tick_params(axis="x", which="both", labelsize=x_tick_font)
    plt.savefig(os.path.join(save_path, f"{fig_name}"), dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_bar_performance_drop(
        data_df,
        figsize,
        dataset,
        title,
        fig_name,
        label_font=21,
        tick_font=20,
        title_font=25,
        legend_font=15,
        save_path=None,
        legend=None,
        legend_pos="upper left",
        dpi=500,
        width=0.25,
        x_label_rotation=0,
        y_label_rotation=0
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    label = data_df["topk"]
    x = np.arange(len(label))
    ax.bar(x - width, data_df["drop_moie"], width=width, label="acc_drop_moie", edgecolor="black")
    ax.bar(x, data_df["drop_pcbm"], width=width, label="acc_drop_cbm", edgecolor="black")
    ax.bar(x + width, data_df["drop_cbm"], width=width, label="acc_drop_pcbm", edgecolor="black")
    ax.set_ylabel("% drop in AUROC", fontsize=label_font, labelpad=12)
    ax.set_xlabel("Concepts Intervened", fontsize=label_font, labelpad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(label)

    handles, _ = ax.get_legend_handles_labels()
    plt.legend(legend, loc=legend_pos, fontsize=legend_font)

    ax.tick_params(axis="x", labelrotation=x_label_rotation, labelsize=tick_font)
    ax.tick_params(axis="y", labelrotation=y_label_rotation, labelsize=tick_font)
    plt.title(title, fontweight="bold", fontsize=title_font)
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, fig_name), dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_bar_completeness(
        data_df,
        figsize,
        dataset,
        title,
        fig_name,
        label_font=24,
        tick_font=20,
        title_font=27,
        legend_font=13.5,
        x_label_rotation=0,
        y_label_rotation=0,
        dpi=500,
        width=0.25,
        save_path=None,
        legends=["MoIE-CXR", "PCBM + ELL"]
):
    # convert that into a dataframe
    # show the dictionary

    # create the base axis
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    label = data_df["topk"]
    x = np.arange(len(label))
    rect1 = ax.bar(x - width, data_df["competeness_moie"], width=width, label="competeness_moie", edgecolor="black")
    rect2 = ax.bar(x, data_df["competeness_pcbm"], width=width, label="competeness_pcbm", edgecolor="black")
    ax.set_ylabel("Completeness score (0-1)", fontsize=label_font, labelpad=12)
    ax.set_xlabel("Concepts", fontsize=label_font, labelpad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(label)

    handles, _ = ax.get_legend_handles_labels()
    plt.legend(legends, loc="upper left", fontsize=legend_font)
    ax.tick_params(axis="x", labelrotation=x_label_rotation, labelsize=tick_font)
    ax.tick_params(axis="y", labelrotation=y_label_rotation, labelsize=tick_font)
    plt.title(title, fontweight="bold", fontsize=title_font)
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, fig_name), dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_acc_tti(
        _topK,
        _acc_g_pcbm,
        _acc_g_cbm,
        _acc_g_moie,
        fig_name,
        title,
        fig_size,
        label_font=24,
        tick_font=20,
        title_font=27,
        legend_font=13.5,
        save_path=None,
        dpi=500,
        y_header="Accuracy %",
        legend="lower right",
        legend_title=["MoIE-CXR", "CBM + ELL", "PCBM + ELL"]
):
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.set_ylabel(f"{y_header}", fontsize=label_font, labelpad=12)
    ax.set_xlabel("Concepts Intervened", fontsize=label_font, labelpad=12)
    ax.tick_params(axis="x", which="both", labelsize=tick_font)
    ax.tick_params(axis="y", which="both", labelsize=tick_font)
    ax.set_xticks(_topK)
    ax.set_xticklabels(_topK)

    plt.title(title, fontweight="bold", fontsize=title_font)

    plt.plot(_topK, _acc_g_moie, marker="*", color="r")
    plt.plot(_topK, _acc_g_cbm, marker="*", color="b")
    plt.plot(_topK, _acc_g_pcbm, marker="*", color="tab:green")

    plt.legend(
        legend_title, loc=legend, fontsize=legend_font
    )
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, fig_name), dpi=dpi)
    plt.show()


def plot_performance_df(
        sample_size,
        moie_no_ft,
        moie_ft,
        moie_r_no_ft,
        moie_r_ft,
        bb,
        fig_name,
        title,
        fig_size,
        label_font=24,
        tick_font=20,
        title_font=27,
        legend_font=13.5,
        save_path=None,
        dpi=500,
        y_header="Accuracy %",
        legend="lower right",
        category_names=[],
        line_width=1,
        marker_size=20
):
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.set_ylabel(f"{y_header}", fontsize=label_font, labelpad=12)
    ax.set_xlabel("% of training samples", fontsize=label_font, labelpad=12)
    ax.tick_params(axis="x", which="both", labelsize=tick_font)
    ax.tick_params(axis="y", which="both", labelsize=tick_font)
    ax.set_xticks(sample_size)
    ax.set_xticklabels(sample_size)
    plt.title(title, fontweight="bold", fontsize=title_font)

    plt.plot(sample_size, moie_no_ft, marker="*", color="tab:red", markersize=marker_size, linewidth=line_width)
    plt.plot(sample_size, moie_ft, marker="*", color="tab:blue", markersize=marker_size, linewidth=line_width)
    plt.plot(sample_size, moie_r_no_ft, marker="*", color="orange", markersize=marker_size, linewidth=line_width)
    plt.plot(sample_size, moie_r_ft, marker="*", color="green", markersize=marker_size, linewidth=line_width)
    plt.plot(sample_size, bb, marker="*", color="gray", markersize=marker_size, linewidth=line_width)

    plt.legend(category_names, loc=legend, fontsize=legend_font)
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, fig_name), dpi=dpi)
    plt.show()
