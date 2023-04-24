from factuality.common.config import Config
from factuality.data.vitamin_c.readers.vitamin_c_stats import VitaminCStats


def main():
    config = Config.instance()
    vcstats = VitaminCStats()
    vcstats.all_sentences()

if __name__ == '__main__':
    main()
