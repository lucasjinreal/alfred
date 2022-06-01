"""
print github trend
"""

# Standard library imports

# Third party imports

try:
    from rich.logging import RichHandler
    import http.client
    from gtrending import fetch_repos
    import colorama
    from click import secho
    import requests
    import os
    import re
    import json
    from random import randint
    import logging
    from time import sleep
    from datetime import datetime, timedelta
    import textwrap
    import math
    from shutil import get_terminal_size
    from rich.align import Align
    from rich.console import Console
    from rich.console import group as render_group
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text
    from rich.panel import Panel
    from rich.columns import Columns
    from xdg import xdg_cache_home

    console = Console()

    FORMAT = "%(message)s"
    httpclient_logger = logging.getLogger("http.client")
except ImportError as e:
    pass

# could be made into config option in the future
try:
    CACHED_RESULT_PATH = xdg_cache_home() / "starcli.json"
except Exception as e:
    CACHED_RESULT_PATH = "/tmp/starcli.json"
print(f"temp cached path: {CACHED_RESULT_PATH}")
CACHE_EXPIRATION = 1  # Minutes
API_URL = "https://api.github.com/search/repositories"

date_range_map = {"today": "daily", "this-week": "weekly", "this-month": "monthly"}

status_actions = {
    "retry": "Failed to retrieve data. Retrying in ",
    "invalid": "The server was unable to process the request.",
    "unauthorized": "The server did not accept the credentials. See: https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token",
    "not_found": "The server indicated no data was found.",
    "unsupported": "The request is not supported.",
    "unknown": "An unknown error occurred.",
    "valid": "The request returned successfully, but an unknown exception occurred.",
}


def httpclient_logging_debug(level=logging.DEBUG):
    def httpclient_log(*args):
        httpclient_logger.log(level, " ".join(args))

    http.client.print = httpclient_log
    http.client.HTTPConnection.debuglevel = 1


def debug_requests_on():
    """Turn on the logging for requests"""

    logging.basicConfig(
        level=logging.DEBUG,
        format=FORMAT,
        datefmt="[%Y-%m-%d]",
        handlers=[RichHandler()],
    )
    logger = logging.getLogger(__name__)

    from http.client import HTTPConnection

    httpclient_logging_debug()

    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True


def convert_datetime(date, date_format="%Y-%m-%d"):
    """Safely convert a date string to datetime"""
    try:
        # try to turn the string into a date-time object
        tmp_date = datetime.strptime(date, date_format)
    except ValueError:  # ValueError will be thrown if format is invalid
        secho(
            "Invalid date: " + date + " must be yyyy-mm-dd",
            fg="bright_red",
        )
        return None
    return tmp_date


def get_date(date):
    """Finds the date info in a string"""
    prefix = ""
    if any(i in date[0] for i in [">", "=", "<"]):
        if "=" in date[1]:
            prefix = date[:2]
            date = date.strip(prefix)
        else:
            prefix = date[0]
            date = date.strip(prefix)
    tmp_date = convert_datetime(date)
    if not tmp_date:
        return None
    return prefix + tmp_date.strftime("%Y-%m-%d")


def get_valid_request(url, auth=""):
    """
    Provide a URL to submit a GET request for and handle a connection error.
    """
    while True:
        try:
            session = requests.Session()
            if auth:
                session.auth = (auth.split(":")[0], auth.split(":")[1])
            request = session.get(url)
        except requests.exceptions.ConnectionError:
            secho("Internet connection error...", fg="bright_red")
            return None

        if not request.status_code in (200, 202):
            handling_code = search_error(request.status_code)
            if handling_code == "retry":
                for i in range(15, 0, -1):
                    secho(
                        f"{status_actions[handling_code]} {i} seconds...",
                        fg="bright_yellow",
                    )  # Print and update a timer

                    sleep(1)
            elif handling_code in status_actions:
                secho(status_actions[handling_code], fg="bright_yellow")
                return None
            else:
                secho("An invalid handling code was returned.", fg="bright_red")
                return None
        else:
            break

    return request


def search_error(status_code):
    """
    This returns a directive on how to handle a given HTTP status code.
    """
    int_status_code = int(
        status_code
    )  # Need to make sure the status code is an integer

    http_code_handling = {
        "200": "valid",
        "202": "valid",
        "204": "valid",
        "400": "invalid",
        "401": "unauthorized",
        "403": "retry",
        "404": "not_found",
        "405": "invalid",
        "422": "not_found",
        "500": "invalid",
        "501": "invalid",
    }

    try:
        return http_code_handling[str(int_status_code)]
    except KeyError:
        return "unsupported"


def search(
    language=None,
    created=None,
    pushed=None,
    stars=">=100",
    topics=[],
    user=None,
    debug=False,
    order="desc",
    auth="",
):
    """Returns repositories searched from GitHub API"""
    date_format = "%Y-%m-%d"  # date format in iso format
    if debug:
        debug_requests_on()
        logger = logging.getLogger(__name__)
        logger.debug("Search: created param:" + created)
        logger.debug("Search: order param: " + order)

    day_range = 0 - randint(100, 400)  # random negative from 100 to 400

    if not created:  # if created not provided
        # creation date: the time now minus a random number of days
        # 100 to 400 days - which was stored in day_range
        created_str = ">=" + (datetime.utcnow() + timedelta(days=day_range)).strftime(
            date_format
        )
    else:  # if created is provided
        created_str = get_date(created)
        if not created_str:
            return None

    if not pushed:  # if pushed not provided
        # pushed date: start, is the time now minus a random number of days
        # 100 to 400 days - which was stored in day_range
        pushed_str = ">=" + (datetime.utcnow() + timedelta(days=day_range)).strftime(
            date_format
        )
    else:  # if pushed is provided
        pushed_str = get_date(pushed)
        if not pushed_str:
            return None

    if user:
        query = f"user:{user}+"
    else:
        query = ""

    query += f"stars:{stars}+created:{created_str}"  # construct query
    query += f"+pushed:{pushed_str}"  # add pushed info to query
    # add language to query
    query += f"+language:{language}" if language else ""
    query += f"".join(["+topic:" + i for i in topics])  # add topics to query

    # use query to construct url
    url = f"{API_URL}?q={query}&sort=stars&order={order}"
    if debug:
        logger.debug("Search: url:" + url)  # print the url when debugging
    if debug and auth:
        logger.debug("Auth: on")
    elif debug:
        logger.debug("Auth: off")

    request = get_valid_request(url, auth)
    if request is None:
        return request

    return request.json()["items"]


def search_github_trending(
    language=None, spoken_language=None, order="desc", stars=">=10", date_range=None
):
    """Returns trending repositories from github trending page"""
    if date_range:
        gtrending_repo_list = fetch_repos(
            language, spoken_language, date_range_map[date_range]
        )
    else:
        gtrending_repo_list = fetch_repos(language, spoken_language)
    repositories = []
    for gtrending_repo in gtrending_repo_list:
        repo_dict = convert_repo_dict(gtrending_repo)
        repo_dict["date_range"] = (
            str(repo_dict["date_range"]) + " stars " + date_range.replace("-", " ")
            if date_range
            else None
        )
        repo_dict["watchers_count"] = -1  # watchers count not available
        # filter by number of stars
        num = [int(s) for s in re.findall(r"\d+", stars)][0]
        if (
            ("<" in stars and repo_dict["stargazers_count"] < num)
            or ("<=" in stars and repo_dict["stargazers_count"] <= num)
            or (">" in stars and repo_dict["stargazers_count"] > num)
            or (">=" in stars and repo_dict["stargazers_count"] >= num)
        ):
            repositories.append(repo_dict)

    if order == "asc":
        return sorted(repositories, key=lambda repo: repo["stargazers_count"])
    return sorted(repositories, key=lambda repo: repo["stargazers_count"], reverse=True)


def convert_repo_dict(gtrending_repo):
    repo_dict = {}
    repo_dict["full_name"] = gtrending_repo.get("fullname")
    repo_dict["name"] = gtrending_repo.get("name")
    repo_dict["html_url"] = gtrending_repo.get("url")
    repo_dict["stargazers_count"] = gtrending_repo.get("stars", -1)
    repo_dict["language"] = gtrending_repo.get("language")
    # gtrending_repo has key `description` and value is empty string if it's empty
    repo_dict["description"] = (
        gtrending_repo.get("description")
        if gtrending_repo.get("description") != ""
        else None
    )
    repo_dict["date_range"] = gtrending_repo.get("currentPeriodStars")
    return repo_dict


def shorten_count(number):
    """Shortens number"""
    if number < 1000:
        return str(number)

    number = int(number)
    new_number = math.ceil(round(number / 100.0, 1)) * 100

    if new_number % 1000 == 0:
        return str(new_number)[0] + "k"
    if new_number < 1000:
        # returns the same old integer if no changes were made
        return str(number)
    else:
        # returns a new string if the number was shortened
        return str(new_number / 1000.0) + "k"


def get_stats(repo):
    """return formatted string of repo stats"""
    stats = f"{repo['stargazers_count']} ⭐ " if repo["stargazers_count"] != "-1" else ""
    stats += f"{repo['watchers_count']} 👀 " if repo["watchers_count"] != "-1" else ""
    return stats


def list_layout(repos):
    """Displays repositories in list layout using rich"""

    LAYOUT_WIDTH = 80

    @render_group()
    def render_repo(repo):
        """Yields renderables for a single repo."""
        yield Rule(style="bright_yellow")
        yield ""
        # Table with description and stats
        title_table = Table.grid(padding=(0, 1))
        title_table.expand = True
        stats = get_stats(repo)
        title = Text(repo["full_name"], overflow="fold")
        title.stylize(f"yellow link {repo['html_url']}")
        date_range_col = (
            Text(
                ("(+" + repo["date_range"] + ")").replace("stars ", ""),
                style="bold cyan",
            )
            if "date_range" in repo.keys() and repo["date_range"]
            else Text("")
        )
        title_table.add_row(title, Text(stats, style="bold blue") + date_range_col)
        title_table.columns[1].no_wrap = True
        title_table.columns[1].justify = "right"
        yield title_table
        yield ""
        lang_table = Table.grid(padding=(0, 1))
        lang_table.expand = True
        language_col = (
            Text(repo["language"], style="bold cyan")
            if repo["language"]
            else Text("unknown language")
        )
        lang_table.add_row(language_col)
        yield lang_table
        yield ""
        # Descripion
        description = repo["description"]
        if description:
            yield Text(description.strip(), style="green")
        else:
            yield "[i green]no description"
        yield ""

    def column(renderable):
        """Constrain width and align to center to create a column."""
        return Align.center(renderable, width=LAYOUT_WIDTH, pad=False)

    for repo in repos:
        console.print(column(render_repo(repo)))
    console.print(column(Rule(style="bright_yellow")))


def table_layout(repos):
    """Displays repositories in a table format using rich"""

    table = Table(leading=1)

    # make the columns
    table.add_column("Name", style="bold cyan")
    table.add_column("Language", style="green")
    table.add_column("Description", style="blue")
    table.add_column("Stats", style="magenta")

    for repo in repos:
        stats = get_stats(repo)
        stats += (
            "\n" + repo["date_range"].replace("stars", "⭐")
            if "date_range" in repo.keys() and repo["date_range"]
            else ""
        )

        if not repo["language"]:  # if language is not provided
            repo["language"] = "None"  # make it a string
        if not repo["description"]:  # same here
            repo["description"] = "None"

        table.add_row(
            repo["full_name"] + "\n" + repo["html_url"],
            repo["language"],  # so that it can work here
            repo["description"],
            stats,
        )
    console.print(table)


def grid_layout(repos):
    """Displays repositories in a grid format using rich"""

    max_desc_len = 90

    panels = []
    for repo in repos:

        stats = get_stats(repo)
        # '\n' added here as it would group both text and new line together
        # hence if date_range isn't present the new line will also not be displayed
        date_range_str = (
            repo["date_range"].replace("stars", "⭐") + "\n"
            if "date_range" in repo.keys() and repo["date_range"]
            else ""
        )

        if not repo["language"]:  # if language is not provided
            repo["language"] = "None"  # make it a string
        if not repo["description"]:
            repo["description"] = "None"

        name = Text(repo["name"], style="bold yellow")
        language = Text(repo["language"], style="magenta")
        description = Text(repo["description"], style="green")
        stats = Text(stats, style="blue")

        # truncate rest of the description if
        # it's more than 90 (max_desc_len) chars
        # using truncate() is better than textwrap
        # because it also takes care of asian characters
        description.truncate(max_desc_len, overflow="ellipsis")

        repo_summary = Text.assemble(
            name,
            "\n",
            stats,
            "\n",
            date_range_str,
            language,
            "\n",
            description,
        )
        panels.append(Panel(repo_summary, expand=True))

    console.print((Columns(panels, width=30, expand=True)))


def print_results(*args, page=False, layout=""):
    """Use a specified layout to print or page the fetched results"""
    if page:
        with console.pager():
            print_layout(layout=layout, *args)
    else:
        print_layout(
            layout=layout,
            *args,
        )


def print_layout(*args, layout="list"):
    if layout == "table":
        table_layout(*args)
    elif layout == "grid":
        grid_layout(*args)
    else:
        list_layout(*args)
    return


def get_github_trending(
    lang="",
    spoken_language="",
    created="",
    topic=[],
    pushed="",
    layout="table",
    stars=">=100",
    limit_results=50,
    order="desc",
    long_stats=True,
    # date_range='today',
    date_range="this-week",
    user="",
    debug=False,
    auth="",
    pager=False,
):
    """Find trending repos on GitHub"""
    if debug:
        import logging

        debug_requests_on()

    tmp_repos = None
    options_key = "{lang}_{spoken_language}_{created}_{topic}_{pushed}_{stars}_{order}_{date_range}_{user}".format(
        lang=lang,
        spoken_language=spoken_language,
        created=created,
        topic=topic,
        pushed=pushed,
        stars=stars,
        order=order,
        date_range=date_range,
        user=user,
    )

    if os.path.exists(CACHED_RESULT_PATH):
        with open(CACHED_RESULT_PATH, "r") as f:
            json_file = json.load(f)
            result = json_file.get(options_key)
            if result:
                t = result[-1].get("time")
                time = datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f")
                diff = datetime.now() - time
                if diff < timedelta(minutes=CACHE_EXPIRATION):
                    if debug:
                        logger = logging.getLogger(__name__)
                        logger.debug("Fetching results from cache")

                    tmp_repos = result

    if not tmp_repos:  # If cache expired or results not yet cached
        if auth and not re.search(".:.", auth):  # Check authentication format
            print(
                f"Invalid authentication format: {auth} must be 'username:token'",
            )
            print(
                "Use --help or see: https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token",
            )
            auth = None

        if (
            not spoken_language and not date_range
        ):  # if filtering by spoken language and date range not required
            tmp_repos = search(
                lang, created, pushed, stars, topic, user, debug, order, auth
            )
        else:
            tmp_repos = search_github_trending(
                lang, spoken_language, order, stars, date_range
            )

        if not tmp_repos:  # if search() returned None
            return
        else:  # Cache results
            tmp_repos.append({"time": str(datetime.now())})
            with open(CACHED_RESULT_PATH, "a+") as f:
                if os.path.getsize(CACHED_RESULT_PATH) == 0:  # file is empty
                    result_dict = {options_key: tmp_repos}
                    f.write(json.dumps(result_dict, indent=4))
                else:  # file is not empty
                    f.seek(0)
                    result_dict = json.load(f)
                    result_dict[options_key] = tmp_repos
                    f.truncate(0)
                    f.write(json.dumps(result_dict, indent=4))
    tmp_repos = [r for r in tmp_repos if "name" in r.keys() and "full_name" in r.keys()]
    repos = tmp_repos[0:limit_results]

    if not long_stats:  # shorten the stat counts when not --long-stats
        for repo in repos:
            repo["stargazers_count"] = shorten_count(repo["stargazers_count"])
            repo["watchers_count"] = shorten_count(repo["watchers_count"])
            if "date_range" in repo.keys() and repo["date_range"]:
                num_stars = repo["date_range"].split()[0]
                repo["date_range"] = repo["date_range"].replace(
                    num_stars, str(shorten_count(int(num_stars.replace(",", ""))))
                )

    print_results(repos, page=pager, layout=layout)
