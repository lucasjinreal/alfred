#!/usr/bin/env python
#
# Copyright (c) 2020 JinTian.
#
# This file is part of alfred
# (see http://jinfagang.github.io).
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# encoding: utf-8

import argparse
import fnmatch
import logging
import os
import sys
from shutil import copyfile
from string import Template
from datetime import datetime
from alfred.utils.log import logger
import regex as re

LOGGER = logger

type_settings = {
    "java": {
        "extensions": [".java", ".scala", ".groovy", ".jape", ".js"],
        "keepFirst": None,
        "blockCommentStartPattern": re.compile(r'^\s*/\*'),
        "blockCommentEndPattern": re.compile(r'\*/\s*$'),
        "lineCommentStartPattern": re.compile(r'\s*//'),
        "lineCommentEndPattern": None,
        "headerStartLine": "/*\n",
        "headerEndLine": " */\n",
        "headerLinePrefix": " * ",
        "headerLineSuffix": None,
    },
    "script": {
        "extensions": [".sh", ".csh", ".py", ".pl"],
        "keepFirst": re.compile(r'^#!|^# -\*-'),
        "blockCommentStartPattern": None,
        "blockCommentEndPattern": None,
        "lineCommentStartPattern": re.compile(r'\s*#'),
        "lineCommentEndPattern": None,
        "headerStartLine": "##\n",
        "headerEndLine": "##\n",
        "headerLinePrefix": "## ",
        "headerLineSuffix": None
    },
    "perl": {
        "extensions": [".pl"],
        "keepFirst": re.compile(r'^#!|^# -\*-'),
        "blockCommentStartPattern": None,
        "blockCommentEndPattern": None,
        "lineCommentStartPattern": re.compile(r'\s*#'),
        "lineCommentEndPattern": None,
        "headerStartLine": "##\n",
        "headerEndLine": "##\n",
        "headerLinePrefix": "## ",
        "headerLineSuffix": None
    },
    "python": {
        "extensions": [".py"],
        "keepFirst": re.compile(r'^#!|^# +pylint|^# +-\*-|^# +coding|^# +encoding'),
        "blockCommentStartPattern": None,
        "blockCommentEndPattern": None,
        "lineCommentStartPattern": re.compile(r'\s*#'),
        "lineCommentEndPattern": None,
        "headerStartLine": "#\n",
        "headerEndLine": "#\n",
        "headerLinePrefix": "# ",
        "headerLineSuffix": None
    },
    "robot": {
        "extensions": [".robot"],
        "keepFirst": re.compile(r'^#!|^# +pylint|^# +-\*-|^# +coding|^# +encoding'),
        "blockCommentStartPattern": None,
        "blockCommentEndPattern": None,
        "lineCommentStartPattern": re.compile(r'\s*#'),
        "lineCommentEndPattern": None,
        "headerStartLine": None,
        "headerEndLine": None,
        "headerLinePrefix": "# ",
        "headerLineSuffix": None
    },
    "xml": {
        "extensions": [".xml"],
        "keepFirst": re.compile(r'^\s*<\?xml.*\?>'),
        "blockCommentStartPattern": re.compile(r'^\s*<!--'),
        "blockCommentEndPattern": re.compile(r'-->\s*$'),
        "lineCommentStartPattern": None,
        "lineCommentEndPattern": None,
        "headerStartLine": "<!--\n",
        "headerEndLine": "  -->\n",
        "headerLinePrefix": "-- ",
        "headerLineSuffix": None
    },
    "sql": {
        "extensions": [".sql"],
        "keepFirst": None,
        "blockCommentStartPattern": None,  # re.compile('^\s*/\*'),
        "blockCommentEndPattern": None,  # re.compile(r'\*/\s*$'),
        "lineCommentStartPattern": re.compile(r'\s*--'),
        "lineCommentEndPattern": None,
        "headerStartLine": "--\n",
        "headerEndLine": "--\n",
        "headerLinePrefix": "-- ",
        "headerLineSuffix": None
    },
    "c": {
        "extensions": [".c", ".cc", ".cpp", "c++", ".h", ".hpp"],
        "keepFirst": None,
        "blockCommentStartPattern": re.compile(r'^\s*/\*'),
        "blockCommentEndPattern": re.compile(r'\*/\s*$'),
        "lineCommentStartPattern": re.compile(r'\s*//'),
        "lineCommentEndPattern": None,
        "headerStartLine": "/*\n",
        "headerEndLine": " */\n",
        "headerLinePrefix": " * ",
        "headerLineSuffix": None
    },
    "ruby": {
        "extensions": [".rb"],
        "keepFirst": "^#!",
        "blockCommentStartPattern": re.compile('^=begin'),
        "blockCommentEndPattern": re.compile(r'^=end'),
        "lineCommentStartPattern": re.compile(r'\s*#'),
        "lineCommentEndPattern": None,
        "headerStartLine": "##\n",
        "headerEndLine": "##\n",
        "headerLinePrefix": "## ",
        "headerLineSuffix": None
    },
    "csharp": {
        "extensions": [".cs"],
        "keepFirst": None,
        "blockCommentStartPattern": None,
        "blockCommentEndPattern": None,
        "lineCommentStartPattern": re.compile(r'\s*//'),
        "lineCommentEndPattern": None,
        "headerStartLine": None,
        "headerEndLine": None,
        "headerLinePrefix": "// ",
        "headerLineSuffix": None
    },
    "vb": {
        "extensions": [".vb"],
        "keepFirst": None,
        "blockCommentStartPattern": None,
        "blockCommentEndPattern": None,
        "lineCommentStartPattern": re.compile(r"^\s*\'"),
        "lineCommentEndPattern": None,
        "headerStartLine": None,
        "headerEndLine": None,
        "headerLinePrefix": "' ",
        "headerLineSuffix": None
    },
    "erlang": {
        "extensions": [".erl", ".src", ".config", ".schema"],
        "keepFirst": None,
        "blockCommentStartPattern": None,
        "blockCommentEndPattern": None,
        "lineCommentStartPattern": None,
        "lineCommentEndPattern": None,
        "headerStartLine": "%% -*- erlang -*-\n%% %CopyrightBegin%\n%%\n",
        "headerEndLine": "%%\n%% %CopyrightEnd%\n\n",
        "headerLinePrefix": "%% ",
        "headerLineSuffix": None,
    }
}

years_pattern = re.compile(
    r"(?<=Copyright\s*(?:\(\s*[Cc©]\s*\)\s*))?([0-9][0-9][0-9][0-9](?:-[0-9][0-9]?[0-9]?[0-9]?)?)",
    re.IGNORECASE)
licensePattern = re.compile(r"license", re.IGNORECASE)
emptyPattern = re.compile(r'^\s*$')

# maps each extension to its processing type. Filled from tpeSettings during initialization
ext2type = {}
patterns = []


# class for dict args. Use --argname key1=val1,val2 key2=val3 key3=val4, val5
class DictArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        dict_args = {}
        if not isinstance(values, (list,)):
            values = (values,)
        for value in values:
            n, v = value.split("=")
            if n not in type_settings:
                LOGGER.error("No valid language '%s' to add additional file extensions for" % n)
            if v and "," in str(v):
                dict_args[n] = v.split(",")
            else:
                dict_args[n] = list()
                dict_args[n].append(str(v).strip())
        setattr(namespace, self.dest, dict_args)


def get_paths(fnpatterns, start_dir="."):
    """
    Retrieve files that match any of the glob patterns from the start_dir and below.
    :param fnpatterns: the file name patterns
    :param start_dir: directory where to start searching
    :return: generator that returns one path after the other
    """
    seen = set()
    for root, dirs, files in os.walk(start_dir):
        names = []
        for pattern in fnpatterns:
            names += fnmatch.filter(files, pattern)
        for name in names:
            path = os.path.join(root, name)
            if path in seen:
                continue
            seen.add(path)
            yield path


def read_template(template_file, vardict, safe_subst=False):
    """
    Read a template file replace variables from the dict and return the lines.
    Throws exception if a variable cannot be replaced.
    :param template_file: template file with variables
    :param vardict: dictionary to replace variables with values
    :param safe_subst:
    :return: lines of the template, with variables replaced
    """
    with open(template_file, 'r') as f:
        lines = f.readlines()
    if safe_subst:
        lines = [Template(line).safe_substitute(vardict) for line in lines]
    else:
        lines = [Template(line).substitute(vardict) for line in lines]
    return lines


def for_type(templatelines, ftype):
    """
    Format the template lines for the given ftype.
    :param templatelines: the lines of the template text
    :param ftype: file type
    :return: header lines
    """
    lines = []
    settings = type_settings[ftype]
    header_start_line = settings["headerStartLine"]
    header_end_line = settings["headerEndLine"]
    header_line_prefix = settings["headerLinePrefix"]
    header_line_suffix = settings["headerLineSuffix"]
    if header_start_line is not None:
        lines.append(header_start_line)
    for line in templatelines:
        tmp = line
        if header_line_prefix is not None and line == '\n':
            tmp = header_line_prefix.rstrip() + tmp
        elif header_line_prefix is not None:
            tmp = header_line_prefix + tmp
        if header_line_suffix is not None:
            tmp = tmp + header_line_suffix
        lines.append(tmp)
    if header_end_line is not None:
        lines.append(header_end_line)
    return lines


##
def read_file(file, encoding='utf-8'):
    """
    Read a file and return a dictionary with the following elements:
    :param file: the file to read
    :param encoding: the options specified by the user
    :return: a dictionary with the following entries or None if the file is not supported:
      - skip: number of lines at the beginning to skip (always keep them when replacing or adding something)
       can also be seen as the index of the first line not to skip
      - headStart: index of first line of detected header, or None if non header detected
      - headEnd: index of last line of detected header, or None
      - yearsLine: index of line which contains the copyright years, or None
      - haveLicense: found a line that matches a pattern that indicates this could be a license header
      - settings: the type settings
    """
    skip = 0
    head_start = None
    head_end = None
    years_line = None
    have_license = False
    extension = os.path.splitext(file)[1]
    LOGGER.debug("File extension is %s", extension)
    # if we have no entry in the mapping from extensions to processing type, return None
    ftype = ext2type.get(extension)
    logging.debug("Type for this file is %s", ftype)
    if not ftype:
        return None
    settings = type_settings.get(ftype)
    with open(file, 'r', encoding=encoding) as f:
        lines = f.readlines()
    # now iterate throw the lines and try to determine the various indies
    # first try to find the start of the header: skip over shebang or empty lines
    keep_first = settings.get("keepFirst")
    block_comment_start_pattern = settings.get("blockCommentStartPattern")
    block_comment_end_pattern = settings.get("blockCommentEndPattern")
    line_comment_start_pattern = settings.get("lineCommentStartPattern")
    i = 0
    LOGGER.info("Processing file {} as {}".format(file, ftype))
    for line in lines:
        if i == 0 and keep_first and keep_first.findall(line):
            skip = i + 1
        elif emptyPattern.findall(line):
            pass
        elif block_comment_start_pattern and block_comment_start_pattern.findall(line):
            head_start = i
            break
        elif line_comment_start_pattern and line_comment_start_pattern.findall(line):
            head_start = i
            break
        elif not block_comment_start_pattern and \
                line_comment_start_pattern and \
                line_comment_start_pattern.findall(line):
            head_start = i
            break
        else:
            # we have reached something else, so no header in this file
            # logging.debug("Did not find the start giving up at line %s, line is >%s<",i,line)
            return {"type": ftype,
                    "lines": lines,
                    "skip": skip,
                    "headStart": None,
                    "headEnd": None,
                    "yearsLine": None,
                    "settings": settings,
                    "haveLicense": have_license
                    }
        i = i + 1
    LOGGER.debug("Found preliminary start at {}, i={}, lines={}".format(head_start, i, len(lines)))
    # now we have either reached the end, or we are at a line where a block start or line comment occurred
    # if we have reached the end, return default dictionary without info
    if i == len(lines):
        LOGGER.debug("We have reached the end, did not find anything really")
        return {"type": ftype,
                "lines": lines,
                "skip": skip,
                "headStart": head_start,
                "headEnd": head_end,
                "yearsLine": years_line,
                "settings": settings,
                "haveLicense": have_license
                }
    # otherwise process the comment block until it ends
    if block_comment_start_pattern:
        LOGGER.debug("Found comment start, process until end")
        for j in range(i, len(lines)):
            LOGGER.debug("Checking line {}".format(j))
            if licensePattern.findall(lines[j]):
                have_license = True
            elif block_comment_end_pattern.findall(lines[j]):
                return {"type": ftype,
                        "lines": lines,
                        "skip": skip,
                        "headStart": head_start,
                        "headEnd": j,
                        "yearsLine": years_line,
                        "settings": settings,
                        "haveLicense": have_license
                        }
            elif years_pattern.findall(lines[j]):
                have_license = True
                years_line = j
        # if we went through all the lines without finding an end, maybe we have some syntax error or some other
        # unusual situation, so lets return no header
        LOGGER.debug("Did not find the end of a block comment, returning no header")
        return {"type": ftype,
                "lines": lines,
                "skip": skip,
                "headStart": None,
                "headEnd": None,
                "yearsLine": None,
                "settings": settings,
                "haveLicense": have_license
                }
    else:
        LOGGER.debug("ELSE1")
        for j in range(i, len(lines)):
            if line_comment_start_pattern.findall(lines[j]) and licensePattern.findall(lines[j]):
                have_license = True
            elif not line_comment_start_pattern.findall(lines[j]):
                LOGGER.debug("ELSE2")
                return {"type": ftype,
                        "lines": lines,
                        "skip": skip,
                        "headStart": i,
                        "headEnd": j - 1,
                        "yearsLine": years_line,
                        "settings": settings,
                        "haveLicense": have_license
                        }
            elif years_pattern.findall(lines[j]):
                have_license = True
                years_line = j
        # if we went through all the lines without finding the end of the block, it could be that the whole
        # file only consisted of the header, so lets return the last line index
        LOGGER.debug("RETURN")
        return {"type": ftype,
                "lines": lines,
                "skip": skip,
                "headStart": i,
                "headEnd": len(lines) - 1,
                "yearsLine": years_line,
                "settings": settings,
                "haveLicense": have_license
                }


def make_backup(file, arguments):
    """
    Backup file by copying it to a file with the extension .bak appended to the name.
    :param file: file to back up
    :param arguments: program args, only backs up, if required by an option
    :return:
    """
    if arguments.b:
        LOGGER.info("Backing up file {} to {}".format(file, file + ".bak"))
        copyfile(file, file + ".bak")


def apply_license(owner, proj_n, year=None, url='google.com', files_dir='./', tmpl='apache-2', rm_old=True):
    """
    Apply new license to all files under target dir
    :param year:
    :param owner:
    :param proj_n:
    :param url:
    :param files_dir:
    :param tmpl:
    :param rm_old:
    :return:
    """
    if not year:
        year = datetime.now().year
        logger.info('year not specific, using year: {}'.format(year))
    logger.info('owner: {}'.format(owner))
    logger.info('project name: {}'.format(proj_n))
    logger.info('start apply license under: {}'.format(files_dir))
    additional_extensions = []
    for t in type_settings:
        settings = type_settings[t]
        exts = settings["extensions"]
        # if additional file extensions are provided by the user, they are "merged" here:
        if additional_extensions and t in additional_extensions:
            for aext in additional_extensions[t]:
                LOGGER.debug("Enable custom file extension '%s' for language '%s'" % (aext, t))
                exts.append(aext)

        for ext in exts:
            ext2type[ext] = t
            patterns.append("*" + ext)

    LOGGER.debug("Allowed file patterns %s" % patterns)
    try:
        error = False
        template_lines = None
        start_dir = files_dir
        settings = dict()
        settings["years"] = year
        settings["owner"] = owner
        settings["projectname"] = proj_n
        settings["projecturl"] = url

        # if we have a template name specified, try to get or load the template
        opt_tmpl = tmpl
        LOGGER.info('using license TEMPLATE: {}'.format(tmpl))
        # first get all the names of our own templates
        # for this get first the path of this file
        templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        if not os.path.exists(templates_dir):
            LOGGER.error("can not found templates dir! {}".format(templates_dir))
            exit(-1)
        LOGGER.info("File path: {}".format(os.path.abspath(__file__)))
        # get all the templates in the templates directory
        templates = [f for f in get_paths("*.tmpl", templates_dir)]
        templates = [(os.path.splitext(os.path.basename(t))[0], t) for t in templates]
        # filter by trying to match the name against what was specified
        tmpls = [t for t in templates if opt_tmpl in t[0]]
        # check if one of the matching template names is identical to the parameter, then take that one
        tmpls_eq = [t for t in tmpls if opt_tmpl == t[0]]
        if len(tmpls_eq) > 0:
            tmpls = tmpls_eq
        if len(tmpls) == 1:
            tmpl_name = tmpls[0][0]
            tmpl_file = tmpls[0][1]
            LOGGER.info("Using template {}".format(tmpl_name))
            template_lines = read_template(tmpl_file, settings)
        else:
            if len(tmpls) == 0:
                # check if we can interpret the option as file
                if os.path.isfile(opt_tmpl):
                    LOGGER.info("Using file {}".format(os.path.abspath(opt_tmpl)))
                    template_lines = read_template(os.path.abspath(opt_tmpl), settings)
                else:
                    LOGGER.error("Not a built-in template and not a file, cannot proceed: {}".format(opt_tmpl))
                    LOGGER.error("Built in templates: {}".format(", ".join([t[0] for t in templates])))
                    error = True
            else:
                LOGGER.error("There are multiple matching template names: {}".format([t[0] for t in tmpls]))
                error = True

        if not error:
            # logging.debug("Got template lines: %s",templateLines)
            # now do the actual processing: if we did not get some error, we have a template loaded or
            # no template at all
            # if we have no template, then we will have the years.
            # now process all the files and either replace the years or replace/add the header
            LOGGER.info("Processing directory: {}".format(start_dir))
            LOGGER.info("Patterns: {}".format(patterns))
            paths = get_paths(patterns, start_dir)
            for file in paths:
                LOGGER.debug("Processing file: %s", file)
                finfo = read_file(file)
                if not finfo:
                    LOGGER.debug("File not supported %s", file)
                    continue
                # logging.debug("FINFO for the file: %s", finfo)
                lines = finfo["lines"]
                LOGGER.debug(
                    "Info for the file: headStart=%s, headEnd=%s, haveLicense=%s, skip=%s, len=%s, yearsline=%s",
                    finfo["headStart"], finfo["headEnd"], finfo["haveLicense"], finfo["skip"], len(lines),
                    finfo["yearsLine"])
                # if we have a template: replace or add
                if template_lines:
                    # make_backup(file, arguments)
                    with open(file, 'w', encoding='utf-8') as fw:
                        # if we found a header, replace it
                        # otherwise, add it after the lines to skip
                        head_start = finfo["headStart"]
                        head_end = finfo["headEnd"]
                        have_license = finfo["haveLicense"]
                        ftype = finfo["type"]
                        skip = finfo["skip"]
                        if head_start is not None and head_end is not None and have_license:
                            LOGGER.debug("Replacing header in file {}".format(file))
                            # first write the lines before the header
                            fw.writelines(lines[0:head_start])
                            #  now write the new header from the template lines
                            fw.writelines(for_type(template_lines, ftype))
                            #  now write the rest of the lines
                            fw.writelines(lines[head_end + 1:])
                        else:
                            LOGGER.debug("Adding header to file {}, skip={}".format(file, skip))
                            fw.writelines(lines[0:skip])
                            fw.writelines(for_type(template_lines, ftype))
                            fw.writelines(lines[skip:])
                    # TODO: optionally remove backup if all worked well?
                else:
                    # no template lines, just update the line with the year, if we found a year
                    years_line = finfo["yearsLine"]
                    if years_line is not None:
                        # make_backup(file, arguments)
                        with open(file, 'w', encoding='utf-8') as fw:
                            LOGGER.debug("Updating years in file {} in line {}".format(file, years_line))
                            fw.writelines(lines[0:years_line])
                            fw.write(years_pattern.sub(year, lines[years_line]))
                            fw.writelines(lines[years_line + 1:])
    finally:
        logging.shutdown()
