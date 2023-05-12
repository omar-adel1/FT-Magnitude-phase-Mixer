import logging

logging.basicConfig(level=logging.INFO,filename="Log.og",filemode="w",
                    format="%(asctime)s - %(levelname)s- %(lineno)d - %(message)s")

x = 2

# logging.info(f"the value of x is {x}")
# logging.warning("warning")
# logging.error("error")
# logging.critical("crictical")

try:
    1/0
except ZeroDivisionError as e:
    logging.exception("error")
logger = logging.getLogger("name")
handler =logging.FileHandler('test.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s- line number: %(lineno)d - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("test DSP")
# logger = logging.basicConfig(level=logging.INFO,filename="hello.log",filemode="w",
#                     format="%(asctime)s - %(levelname)s- %(lineno)d - %(message)s")
# logger=logging.info("test dsp")