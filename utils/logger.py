import logging

class Logger:
    def __init__(self, logger_file: str):
        # set logging file
        logging.basicConfig(filename=logger_file, level=logging.INFO, filemode='w')

        # create console processing program
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # add console to logging
        logging.getLogger().addHandler(console_handler)

    def info(self, *args, **kwargs):
        logging.info(*args, **kwargs)

# print = print_to_log

# # 示例print语句
# print("This will be logged and printed to console.")

# # 关闭日志记录器
# logging.shutdown()