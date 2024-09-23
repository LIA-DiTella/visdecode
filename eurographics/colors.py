RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"
BOLD = "\033[1m"

def red(str): return RED + str + RESET
def green(str): return GREEN + str + RESET
def yellow(str): return YELLOW + str + RESET
def blue(str): return BLUE + str + RESET
def magenta(str): return MAGENTA + str + RESET
def cyan(str): return CYAN + str + RESET
def bold(str): return BOLD + str + RESET