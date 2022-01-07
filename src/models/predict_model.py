# -*- coding: utf-8 -*-
import logging
import click

@click.command()
@click.argument("input_filepath_model", type=click.Path(exists=True))
def main(input_filepath_model):

    logger = logging.getLogger(__name__)
    logger.info("predict")

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()