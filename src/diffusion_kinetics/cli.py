import argparse
from diffusion_kinetics.pipeline.pipeline import Pipeline
from diffusion_kinetics.pipeline.multi_pipeline import MultiPipeline
from diffusion_kinetics.pipeline.pipeline_config import SingleProcessPipelineConfig, MultiProcessPipelineConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion Kinetics')
    parser.add_argument('-i', '--input', help='Input file')
    parser.add_argument('-c', '--config', help='Config file')
    parser.add_argument('-o', '--output', help='Output file')
    return parser.parse_args()

def main():
    args = parse_args()
    try:
        pipeline = MultiPipeline(args.input, args.output)
        print("Running multi-process pipeline...")
        pipeline.run(args.config)
    except:
        pipeline = Pipeline(args.input, args.output)
        print("Running single-process pipeline...")
        pipeline.run(args.config)
    
if __name__ == "__main__":
    main()