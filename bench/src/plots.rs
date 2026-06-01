//! SVG plots for Loupe-only benchmark results.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

use crate::filters::{CORRELATION_MODES, Correlation, SELECTIVITY_LEVELS};
use crate::matrix::{CellResult, Strategy};

const WIDTH: f64 = 920.0;
const HEIGHT: f64 = 560.0;
const LEFT: f64 = 90.0;
const RIGHT: f64 = 40.0;
const TOP: f64 = 60.0;
const BOTTOM: f64 = 70.0;

/// The SVG files written for one benchmark run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlotFiles {
    pub fixed_vs_planner: PathBuf,
    pub hard_filter_cells: PathBuf,
    pub acorn_wins: PathBuf,
    pub flat_wins: PathBuf,
}

/// Writes Loupe-only SVG plots to a directory.
pub fn write_loupe_plots(dir: &str, results: &[CellResult]) -> io::Result<PlotFiles> {
    let dir = Path::new(dir);
    std::fs::create_dir_all(dir)?;

    let files = PlotFiles {
        fixed_vs_planner: dir.join("fixed-vs-planner.svg"),
        hard_filter_cells: dir.join("hard-filter-cells.svg"),
        acorn_wins: dir.join("acorn-adaptive-wins.svg"),
        flat_wins: dir.join("flat-wins.svg"),
    };

    std::fs::write(&files.fixed_vs_planner, fixed_vs_planner_svg(results))?;
    std::fs::write(&files.hard_filter_cells, hard_filter_cells_svg(results))?;
    std::fs::write(
        &files.acorn_wins,
        winner_grid_svg(
            results,
            Strategy::AdaptiveAcorn,
            "Where Acorn+AdaptiveEf Wins",
        ),
    )?;
    std::fs::write(
        &files.flat_wins,
        winner_grid_svg(results, Strategy::Flat, "Where Flat Wins"),
    )?;

    Ok(files)
}

/// Renders average recall versus distance work for each runner.
pub fn fixed_vs_planner_svg(results: &[CellResult]) -> String {
    let summaries = strategy_summaries(results);
    let max_distance = summaries
        .iter()
        .map(|summary| summary.distance_computations)
        .fold(1.0, f64::max);

    let mut body = String::new();
    body.push_str(&svg_start("Fixed Strategies vs Planner"));
    body.push_str(&axes("avg distance computations", "avg recall"));

    for summary in summaries {
        let x = LEFT + (summary.distance_computations / max_distance) * plot_width();
        let y = TOP + (1.0 - summary.recall.clamp(0.0, 1.0)) * plot_height();
        body.push_str(&format!(
            "<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"8\" fill=\"{}\" />\n",
            color_for(summary.strategy)
        ));
        body.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" class=\"label\">{} ({:.2})</text>\n",
            x + 12.0,
            y + 4.0,
            escape(strategy_name(summary.strategy)),
            summary.recall
        ));
    }

    body.push_str(&svg_end());
    body
}

/// Renders the hardest anti-correlated cells as a recall heatmap.
pub fn hard_filter_cells_svg(results: &[CellResult]) -> String {
    let rows: Vec<&CellResult> = results
        .iter()
        .filter(|row| row.correlation == Correlation::AntiCorrelated)
        .filter(|row| row.selectivity <= 0.01)
        .collect();
    let mut body = String::new();
    body.push_str(&svg_start("Hard Filter Cells"));
    body.push_str(&format!(
        "<text x=\"{LEFT}\" y=\"42\" class=\"subtitle\">anti-correlated filters at 1% and 0.1% selectivity</text>\n"
    ));

    let runners = [
        Strategy::Flat,
        Strategy::PostFilter,
        Strategy::InTraversal,
        Strategy::Acorn,
        Strategy::AdaptiveAcorn,
        Strategy::Planner,
    ];
    let hard_selectivities = [0.01_f32, 0.001_f32];
    draw_heatmap_header(&mut body, &runners);

    for (row_slot, selectivity) in hard_selectivities.iter().enumerate() {
        let y = 100.0 + row_slot as f64 * 80.0;
        body.push_str(&format!(
            "<text x=\"30\" y=\"{:.1}\" class=\"label\">{:.1}%</text>\n",
            y + 34.0,
            selectivity * 100.0
        ));
        for (col_slot, runner) in runners.iter().enumerate() {
            let recall = rows
                .iter()
                .find(|row| row.strategy == *runner && nearly_equal(row.selectivity, *selectivity))
                .map(|row| row.recall)
                .unwrap_or(0.0);
            draw_heat_cell(&mut body, col_slot, y, recall, recall_text(recall));
        }
    }

    body.push_str(&svg_end());
    body
}

/// Renders where one runner wins on recall first and distance work second.
pub fn winner_grid_svg(results: &[CellResult], target: Strategy, title: &str) -> String {
    let winners = winner_by_cell(results);
    let mut body = String::new();
    body.push_str(&svg_start(title));
    body.push_str(&format!(
        "<text x=\"{LEFT}\" y=\"42\" class=\"subtitle\">winner uses highest recall, then lowest distance work</text>\n"
    ));

    draw_grid_header(&mut body);
    for (row_slot, correlation) in CORRELATION_MODES.iter().enumerate() {
        let y = 110.0 + row_slot as f64 * 80.0;
        body.push_str(&format!(
            "<text x=\"20\" y=\"{:.1}\" class=\"label\">{}</text>\n",
            y + 36.0,
            escape(correlation_name(*correlation))
        ));
        for (col_slot, selectivity) in SELECTIVITY_LEVELS.iter().enumerate() {
            let winner = winners.get(&cell_key(*selectivity, *correlation));
            let is_target = winner.is_some_and(|strategy| *strategy == target);
            let fill = if is_target {
                color_for(target)
            } else {
                "#f1f5f9"
            };
            let x = LEFT + col_slot as f64 * 145.0;
            body.push_str(&format!(
                "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"120\" height=\"54\" rx=\"6\" fill=\"{fill}\" stroke=\"#cbd5e1\" />\n"
            ));
            let label = winner
                .map(|strategy| strategy_name(*strategy))
                .unwrap_or("none");
            body.push_str(&format!(
                "<text x=\"{:.1}\" y=\"{:.1}\" class=\"cell-label\">{}</text>\n",
                x + 10.0,
                y + 32.0,
                escape(label)
            ));
        }
    }

    body.push_str(&svg_end());
    body
}

#[derive(Clone, Copy)]
struct StrategySummary {
    strategy: Strategy,
    recall: f64,
    distance_computations: f64,
}

fn strategy_summaries(results: &[CellResult]) -> Vec<StrategySummary> {
    let mut sums: BTreeMap<Strategy, (f64, f64, usize)> = BTreeMap::new();
    for row in results {
        let entry = sums.entry(row.strategy).or_insert((0.0, 0.0, 0));
        entry.0 += f64::from(row.recall);
        entry.1 += row.avg_distance_computations;
        entry.2 += 1;
    }

    sums.into_iter()
        .filter_map(|(strategy, (recall, distance, count))| {
            if count == 0 {
                return None;
            }
            Some(StrategySummary {
                strategy,
                recall: recall / count as f64,
                distance_computations: distance / count as f64,
            })
        })
        .collect()
}

fn winner_by_cell(results: &[CellResult]) -> BTreeMap<String, Strategy> {
    let mut winners: BTreeMap<String, &CellResult> = BTreeMap::new();
    for row in results
        .iter()
        .filter(|row| row.strategy != Strategy::Planner)
    {
        let key = cell_key(row.selectivity, row.correlation);
        let replace = winners
            .get(&key)
            .is_none_or(|current| row_beats(row, current));
        if replace {
            winners.insert(key, row);
        }
    }

    winners
        .into_iter()
        .map(|(key, row)| (key, row.strategy))
        .collect()
}

fn row_beats(candidate: &CellResult, current: &CellResult) -> bool {
    let recall_cmp = candidate.recall.total_cmp(&current.recall);
    if recall_cmp == Ordering::Greater {
        return true;
    }
    if recall_cmp == Ordering::Less {
        return false;
    }

    candidate
        .avg_distance_computations
        .total_cmp(&current.avg_distance_computations)
        == Ordering::Less
}

fn draw_heatmap_header(body: &mut String, runners: &[Strategy]) {
    for (slot, runner) in runners.iter().enumerate() {
        let x = LEFT + slot as f64 * 130.0;
        body.push_str(&format!(
            "<text x=\"{x:.1}\" y=\"86\" class=\"label\">{}</text>\n",
            escape(strategy_name(*runner))
        ));
    }
}

fn draw_grid_header(body: &mut String) {
    for (slot, selectivity) in SELECTIVITY_LEVELS.iter().enumerate() {
        let x = LEFT + slot as f64 * 145.0;
        body.push_str(&format!(
            "<text x=\"{x:.1}\" y=\"92\" class=\"label\">{:.1}%</text>\n",
            selectivity * 100.0
        ));
    }
}

fn draw_heat_cell(body: &mut String, col_slot: usize, y: f64, recall: f32, label: String) {
    let x = LEFT + col_slot as f64 * 130.0;
    let intensity = recall.clamp(0.0, 1.0);
    let fill = heat_color(intensity);
    body.push_str(&format!(
        "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"105\" height=\"54\" rx=\"6\" fill=\"{fill}\" stroke=\"#cbd5e1\" />\n"
    ));
    body.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" class=\"cell-label\">{}</text>\n",
        x + 28.0,
        y + 33.0,
        escape(&label)
    ));
}

fn axes(x_label: &str, y_label: &str) -> String {
    let x1 = LEFT;
    let y1 = HEIGHT - BOTTOM;
    let x2 = WIDTH - RIGHT;
    let y2 = TOP;
    format!(
        "<line x1=\"{x1}\" y1=\"{y1}\" x2=\"{x2}\" y2=\"{y1}\" class=\"axis\" />\n\
         <line x1=\"{x1}\" y1=\"{y1}\" x2=\"{x1}\" y2=\"{y2}\" class=\"axis\" />\n\
         <text x=\"{:.1}\" y=\"{:.1}\" class=\"axis-label\">{}</text>\n\
         <text x=\"18\" y=\"{:.1}\" class=\"axis-label\" transform=\"rotate(-90 18,{:.1})\">{}</text>\n",
        WIDTH / 2.0 - 70.0,
        HEIGHT - 20.0,
        escape(x_label),
        HEIGHT / 2.0 + 90.0,
        HEIGHT / 2.0 + 90.0,
        escape(y_label)
    )
}

fn svg_start(title: &str) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{WIDTH:.0}\" height=\"{HEIGHT:.0}\" viewBox=\"0 0 {WIDTH:.0} {HEIGHT:.0}\">\n\
         <style>\
         .title {{ font: 700 24px system-ui, sans-serif; fill: #0f172a; }}\
         .subtitle {{ font: 14px system-ui, sans-serif; fill: #475569; }}\
         .label {{ font: 12px system-ui, sans-serif; fill: #334155; }}\
         .cell-label {{ font: 12px system-ui, sans-serif; fill: #0f172a; }}\
         .axis {{ stroke: #475569; stroke-width: 1.5; }}\
         .axis-label {{ font: 13px system-ui, sans-serif; fill: #334155; }}\
         </style>\n\
         <rect width=\"100%\" height=\"100%\" fill=\"#ffffff\" />\n\
         <text x=\"{LEFT}\" y=\"32\" class=\"title\">{}</text>\n",
        escape(title)
    )
}

fn svg_end() -> String {
    "</svg>\n".to_string()
}

fn plot_width() -> f64 {
    WIDTH - LEFT - RIGHT
}

fn plot_height() -> f64 {
    HEIGHT - TOP - BOTTOM
}

fn cell_key(selectivity: f32, correlation: Correlation) -> String {
    format!("{selectivity:.6}:{}", correlation_name(correlation))
}

fn nearly_equal(left: f32, right: f32) -> bool {
    (left - right).abs() < 0.000_01
}

fn recall_text(recall: f32) -> String {
    format!("{recall:.2}")
}

fn heat_color(value: f32) -> String {
    let clamped = value.clamp(0.0, 1.0);
    let red = 239.0 - clamped * 125.0;
    let green = 246.0 - clamped * 49.0;
    let blue = 255.0 - clamped * 161.0;
    format!("rgb({red:.0},{green:.0},{blue:.0})")
}

fn color_for(strategy: Strategy) -> &'static str {
    match strategy {
        Strategy::Flat => "#2563eb",
        Strategy::PostFilter => "#dc2626",
        Strategy::InTraversal => "#f59e0b",
        Strategy::Acorn => "#7c3aed",
        Strategy::AdaptiveAcorn => "#16a34a",
        Strategy::Planner => "#0f172a",
    }
}

fn strategy_name(strategy: Strategy) -> &'static str {
    match strategy {
        Strategy::Flat => "Flat",
        Strategy::PostFilter => "PostFilter",
        Strategy::InTraversal => "InTraversal",
        Strategy::Acorn => "Acorn",
        Strategy::AdaptiveAcorn => "Acorn+AdaptiveEf",
        Strategy::Planner => "Planner",
    }
}

fn correlation_name(correlation: Correlation) -> &'static str {
    match correlation {
        Correlation::Random => "random",
        Correlation::Positive => "positive",
        Correlation::AntiCorrelated => "anti_correlated",
        Correlation::ClusteredByTenant => "clustered_by_tenant",
    }
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(
        selectivity: f32,
        correlation: Correlation,
        strategy: Strategy,
        recall: f32,
        distance: f64,
    ) -> CellResult {
        CellResult {
            selectivity,
            correlation,
            strategy,
            strategy_chosen: strategy_name(strategy).to_string(),
            recall,
            avg_distance_computations: distance,
            avg_latency_micros: distance / 10.0,
            avg_returned: 10.0,
        }
    }

    fn sample_results() -> Vec<CellResult> {
        let mut rows = Vec::new();
        for correlation in CORRELATION_MODES {
            for selectivity in SELECTIVITY_LEVELS {
                rows.push(row(selectivity, correlation, Strategy::Flat, 1.0, 50.0));
                rows.push(row(
                    selectivity,
                    correlation,
                    Strategy::AdaptiveAcorn,
                    if correlation == Correlation::AntiCorrelated {
                        1.0
                    } else {
                        0.9
                    },
                    if correlation == Correlation::AntiCorrelated {
                        40.0
                    } else {
                        80.0
                    },
                ));
                rows.push(row(selectivity, correlation, Strategy::Planner, 1.0, 45.0));
            }
        }
        rows
    }

    #[test]
    fn renders_fixed_vs_planner_svg() {
        let svg = fixed_vs_planner_svg(&sample_results());

        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("Fixed Strategies vs Planner"));
        assert!(svg.contains("Planner"));
    }

    #[test]
    fn renders_winner_grids() {
        let rows = sample_results();
        let acorn = winner_grid_svg(&rows, Strategy::AdaptiveAcorn, "Where Acorn Wins");
        let flat = winner_grid_svg(&rows, Strategy::Flat, "Where Flat Wins");

        assert!(acorn.contains("Where Acorn Wins"));
        assert!(acorn.contains("Acorn+AdaptiveEf"));
        assert!(flat.contains("Where Flat Wins"));
        assert!(flat.contains("Flat"));
    }

    #[test]
    fn writes_all_plot_files() {
        let dir = tempfile::tempdir().unwrap();
        let files = write_loupe_plots(dir.path().to_str().unwrap(), &sample_results()).unwrap();

        for path in [
            files.fixed_vs_planner,
            files.hard_filter_cells,
            files.acorn_wins,
            files.flat_wins,
        ] {
            let content = std::fs::read_to_string(path).unwrap();
            assert!(content.starts_with("<svg"));
        }
    }
}
