use clap::Parser;

/// Train AlphaTak
#[derive(Parser)]
pub struct Args {
    /// Path to model
    pub model_path: String,
    /// How many virtual rollouts to perform per batch
    #[clap(short, long, default_value_t = 64)]
    pub batch_size: usize,
    /// How many batches to perform in parallel
    #[clap(short, long, default_value_t = 1)]
    pub pipeline_depth: usize,
    /// Path to PTN game file
    #[clap(short, long)]
    pub ptn_file: Option<String>,
    /// Disable GPU usage
    #[clap(short, long)]
    pub no_gpu: bool,
}
