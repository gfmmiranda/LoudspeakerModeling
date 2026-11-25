from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

def _new_axes(figsize):
    figure, axes = plt.subplots(figsize=figsize)
    axes.set_aspect("equal")
    axes.axis("off")
    return figure, axes


def render_block_diagram(diagram, *, figsize=None):
    sum_x = 1.2
    block_gap = 1.05
    block_height = 0.9
    block_widths = {
        "electrical_admittance": 2.3,
        "force_factor": 1.5,
        "mechanical_admittance": 4.0,
        "integration": 1.5,
        "piston_area": 1.5,
        "monopole_radiation": 3.4,
        "back_emf": 1.5,
        "vent_flow": 2.4,
        "vent_radiation": 3.4,
    }

    def width_of(block):
        return block.width if block.width is not None else block_widths[block.id]

    branch = diagram.branch
    forward_width = sum(width_of(block) + block_gap for block in diagram.forward_blocks)
    figure_height = 4
    if branch is not None:
        forward_width += sum(width_of(block) + block_gap for block in branch.blocks)
        figure_height = 6
    if figsize is None:
        figsize = (max(12, 0.9 * forward_width), figure_height)
    figure, axes = _new_axes(figsize)

    axes.add_patch(Circle((sum_x, 0), 0.36, facecolor="white", edgecolor="#20252b", linewidth=1.5))
    axes.text(sum_x, 0, r"$\Sigma$", ha="center", va="center", fontsize=12)
    axes.text(sum_x - 0.17, 0.47, "+", ha="center", va="center", fontsize=11)
    axes.text(sum_x, -0.48, "−", ha="center", va="center", fontsize=11)

    input_signal = diagram.signal(diagram.input_id)
    axes.add_patch(FancyArrowPatch((0, 0), (sum_x - 0.36, 0), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color="#20252b"))
    axes.text(0.42, 0.17, f"{input_signal.label} [{input_signal.unit}]", ha="center", va="bottom", fontsize=10)

    previous_x = sum_x + 0.36
    previous_signal = diagram.signal(diagram.summing_junction.output_signal)
    signal_positions = {diagram.summing_junction.output_signal: previous_x}
    for block in diagram.forward_blocks:
        block_width = width_of(block)
        center_x = previous_x + block_gap + block_width / 2
        left = center_x - block_width / 2
        right = center_x + block_width / 2
        axes.add_patch(Rectangle((left, -block_height / 2), block_width, block_height, facecolor="#f4f1e8", edgecolor="#20252b", linewidth=1.5))
        axes.text(center_x, 0, block.label, ha="center", va="center", fontsize=11)
        axes.add_patch(FancyArrowPatch((previous_x, 0), (left, 0), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color="#20252b"))
        # Keep the label clear of the vertical line where the branch is tapped.
        label_below = branch is not None and previous_signal.id == branch.tap_signal
        axes.text((previous_x + left) / 2, -0.18 if label_below else 0.18, f"{previous_signal.label} [{previous_signal.unit}]", ha="center", va="top" if label_below else "bottom", fontsize=9)
        previous_x = right
        previous_signal = diagram.signal(block.output_signal)
        signal_positions[block.output_signal] = right

    line_color = "#20252b"
    if branch is None:
        output_x = previous_x + 1.25
        axes.add_patch(FancyArrowPatch((previous_x, 0), (output_x, 0), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color=line_color))
        axes.text((previous_x + output_x) / 2, 0.18, f"{previous_signal.label} [{previous_signal.unit}]", ha="center", va="bottom", fontsize=9)
    else:
        branch_y = 2.0
        tap_x = signal_positions[branch.tap_signal] + 0.35
        axes.add_patch(Circle((tap_x, 0), 0.06, facecolor=line_color, edgecolor=line_color))
        axes.plot([tap_x, tap_x], [0, branch_y], color=line_color, linewidth=1.3)

        branch_x = tap_x
        branch_signal = diagram.signal(branch.tap_signal)
        for index, block in enumerate(branch.blocks):
            block_width = width_of(block)
            center_x = branch_x + block_gap + block_width / 2
            left = center_x - block_width / 2
            right = center_x + block_width / 2
            axes.add_patch(Rectangle((left, branch_y - block_height / 2), block_width, block_height, facecolor="#f4f1e8", edgecolor=line_color, linewidth=1.5))
            axes.text(center_x, branch_y, block.label, ha="center", va="center", fontsize=11)
            axes.add_patch(FancyArrowPatch((branch_x, branch_y), (left, branch_y), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color=line_color))
            if index > 0:
                axes.text((branch_x + left) / 2, branch_y + 0.18, f"{branch_signal.label} [{branch_signal.unit}]", ha="center", va="bottom", fontsize=9)
            branch_x = right
            branch_signal = diagram.signal(block.output_signal)

        merge_x = max(previous_x, branch_x) + 1.6
        merge_radius = 0.36

        axes.add_patch(FancyArrowPatch((previous_x, 0), (merge_x - merge_radius, 0), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color=line_color))
        axes.text((previous_x + merge_x - merge_radius) / 2, 0.18, f"{previous_signal.label} [{previous_signal.unit}]", ha="center", va="bottom", fontsize=9)

        axes.plot([branch_x, merge_x], [branch_y, branch_y], color=line_color, linewidth=1.3)
        axes.add_patch(FancyArrowPatch((merge_x, branch_y), (merge_x, merge_radius), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color=line_color))
        axes.text((branch_x + merge_x) / 2, branch_y + 0.18, f"{branch_signal.label} [{branch_signal.unit}]", ha="center", va="bottom", fontsize=9)

        axes.add_patch(Circle((merge_x, 0), merge_radius, facecolor="white", edgecolor=line_color, linewidth=1.5))
        axes.text(merge_x, 0, r"$\Sigma$", ha="center", va="center", fontsize=12)
        axes.text(merge_x - 0.5, 0.2, "+", ha="center", va="center", fontsize=11)
        axes.text(merge_x + 0.2, 0.55, "+", ha="center", va="center", fontsize=11)

        output_x = merge_x + merge_radius + 1.5
        output_signal = diagram.signal(diagram.output_id)
        axes.add_patch(FancyArrowPatch((merge_x + merge_radius, 0), (output_x, 0), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color=line_color))
        axes.text((merge_x + merge_radius + output_x) / 2, 0.18, f"{output_signal.label} [{output_signal.unit}]", ha="center", va="bottom", fontsize=9)

    feedback = diagram.feedback_block
    velocity_x = signal_positions[feedback.input_signal] + 0.35
    feedback_x = velocity_x
    feedback_y = -2.0
    feedback_width = block_widths[feedback.id]
    axes.add_patch(FancyArrowPatch((velocity_x, 0), (feedback_x, feedback_y + block_height / 2), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color="#20252b"))
    velocity_signal = diagram.signal(feedback.input_signal)
    axes.text(velocity_x + 0.15, -0.8, f"{velocity_signal.label} [{velocity_signal.unit}]", ha="left", va="center", fontsize=9)
    axes.add_patch(Rectangle((feedback_x - feedback_width / 2, feedback_y - block_height / 2), feedback_width, block_height, facecolor="#f4f1e8", edgecolor="#20252b", linewidth=1.5))
    axes.text(feedback_x, feedback_y, feedback.label, ha="center", va="center", fontsize=11)
    feedback_signal = diagram.signal(feedback.output_signal)
    axes.add_patch(FancyArrowPatch((feedback_x - feedback_width / 2, feedback_y), (sum_x, -0.36), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color="#20252b", connectionstyle="angle3,angleA=180,angleB=-90"))
    axes.text((feedback_x + sum_x) / 2, feedback_y - 0.18, f"{feedback_signal.label} [{feedback_signal.unit}]", ha="center", va="top", fontsize=9)

    axes.set_xlim(-0.2, output_x + 0.3)
    axes.set_ylim(-2.8, 1.1 if branch is None else 3.0)
    figure.tight_layout()
    return figure, axes