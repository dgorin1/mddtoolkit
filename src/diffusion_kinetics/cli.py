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
    if MultiProcessPipelineConfig.load(args.config):
        pipeline = MultiPipeline(args.input, args.output)
    elif SingleProcessPipelineConfig.load(args.config):
        pipeline = Pipeline(args.input, args.output)
    else :
        raise ValueError(f"config must be a path to a yaml file, a dictionary, or a SingleProcessPipelineConfig object. Got: {config.__class__.__name__}")
    pipeline.run(args.config)
    
if __name__ == "__main__":
    main()