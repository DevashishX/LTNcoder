# Copyright 2018 Timo Nolle
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

from pathlib import Path

import arrow

# Base
ROOT_DIR = Path(__file__).parent.parent

# Base directories
OUT_DIR = ROOT_DIR / '.out'  # For anything that is being generated
RES_DIR = ROOT_DIR / '.res'  # For resources shipped with the repository
CACHE_DIR = OUT_DIR / '.cache'  # Used to cache event logs, results, etc.

# Resources
PROCESS_MODEL_DIR = RES_DIR / 'process_models'  # Randomly generated process models from PLG2
BPIC_DIR = RES_DIR / 'bpic'  # BPIC logs in XES format

# Output
EVENTLOG_DIR = OUT_DIR / 'eventlogs'  # For generated event logs
MODEL_DIR = OUT_DIR / 'models'  # For anomaly detection models
PLOT_DIR = OUT_DIR / 'plots'  # For plots

# Cache
EVENTLOG_CACHE_DIR = CACHE_DIR / 'eventlogs'  # For caching datasets so the event log does not always have to be loaded
RESULT_DIR = CACHE_DIR / 'results'  # For caching anomaly detection results

# Config
CONFIG_DIR = ROOT_DIR / '.config'

# Database
DATABASE_FILE = OUT_DIR / 'april.db'

# Extensions
MODEL_EXT = '.keras'
RESULT_EXT = '.result'

# Misc
DATE_FORMAT = 'YYYYMMDD-HHmmss.SSSSSS'


def generate():
    """Generate directories."""
    dirs = [
        ROOT_DIR,
        OUT_DIR,
        RES_DIR,
        CACHE_DIR,
        RESULT_DIR,
        EVENTLOG_CACHE_DIR,
        MODEL_DIR,
        PROCESS_MODEL_DIR,
        EVENTLOG_DIR,
        BPIC_DIR,
        PLOT_DIR
    ]
    for d in dirs:
        if not d.exists():
            d.mkdir()


def split_eventlog_name(name):
    try:
        s = name.split('-')
        model = s[0]
        p = float(s[1])
        id = int(s[2])
    except Exception:
        model = None
        p = None
        id = None
    return model, p, id


def split_model_name(name):
    try:
        s = name.split('_')
        event_log_name = s[0]
        ad = s[1]
        date = arrow.get(s[2], DATE_FORMAT)
    except Exception as e:
        event_log_name = None
        ad = None
        date = None
    return event_log_name, ad, date


class File(object):
    ext = None

    def __init__(self, path):
        if not isinstance(path, Path):
            path = Path(path)

        self.path = path
        self.file = self.path.name
        self.name = self.path.stem
        self.str_path = str(path)

    def remove(self):
        import os
        if self.path.exists():
            os.remove(self.path)


class EventLogFile(File):
    def __init__(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        if '.json' not in path.suffixes:
            path = Path(str(path) + '.json.gz')
        if not path.is_absolute():
            path = EVENTLOG_DIR / path.name

        super(EventLogFile, self).__init__(path)

        if len(self.path.suffixes) > 1:
            self.name = Path(self.path.stem).stem

        self.model, self.p, self.id = split_eventlog_name(self.name)

    @property
    def cache_file(self):
        return EVENTLOG_CACHE_DIR / (self.name + '.pkl.gz')


class ModelFile(File):
    ext = MODEL_EXT

    def __init__(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        if path.suffix != self.ext:
            path = Path(str(path) + self.ext)
        if not path.is_absolute():
            path = MODEL_DIR / path.name

        super(ModelFile, self).__init__(path)

        self.event_log_name, self.ad, self.date = split_model_name(self.name)
        self.model, self.p, self.id = split_eventlog_name(self.event_log_name)

    @property
    def result_file(self):
        return RESULT_DIR / (self.name + RESULT_EXT)


class ResultFile(File):
    ext = RESULT_EXT

    @property
    def model_file(self):
        return MODEL_DIR / (self.name + MODEL_EXT)


def get_event_log_files(path=None):
    if path is None:
        path = EVENTLOG_DIR
    for f in path.glob('*.json*'):
        yield EventLogFile(f)


def get_model_files(path=None):
    if path is None:
        path = MODEL_DIR
    for f in path.glob(f'*{MODEL_EXT}'):
        yield ModelFile(f)


def get_result_files(path=None):
    if path is None:
        path = RESULT_DIR
    for f in path.glob(f'*{RESULT_EXT}'):
        yield ResultFile(f)


def get_process_model_files(path=None):
    if path is None:
        path = PROCESS_MODEL_DIR
    for f in path.glob('*.plg'):
        yield f.stem


def download_bpic_logs():
    import gzip
    import requests
    from tqdm import tqdm

    logs = [
        ('BPIC12.xes.gz', 'https://data.4tu.nl/ndownloader/files/24027287'),
        ('BPIC13_incidents.xes.gz', 'https://data.4tu.nl/ndownloader/files/24033593'),
        ('BPIC13_open_problems.xes.gz', 'https://data.4tu.nl/ndownloader/files/24026252'),
        ('BPIC13_closed_problems.xes.gz', 'https://data.4tu.nl/ndownloader/files/24070847'),
        ('BPIC14_incidents.csv', 'https://data.4tu.nl/ndownloader/files/24031637'),
        ('BPIC14_interactions.csv', 'https://data.4tu.nl/ndownloader/files/24031670'),
        ('BPIC15_1.xes', 'https://data.4tu.nl/ndownloader/files/24063818'),
        ('BPIC15_2.xes', 'https://data.4tu.nl/ndownloader/files/24044639'),
        ('BPIC15_3.xes', 'https://data.4tu.nl/ndownloader/files/24076154'),
        ('BPIC15_4.xes', 'https://data.4tu.nl/ndownloader/files/24045332'),
        ('BPIC15_5.xes', 'https://data.4tu.nl/ndownloader/files/24069341'),
        ('BPIC16_logged_in.csv.zip', 'https://data.4tu.nl/ndownloader/files/23991602'),
        ('BPIC16_not_logged_in.csv.zip', 'https://data.4tu.nl/ndownloader/files/24063158'),
        ('BPIC16_werkmap.csv', 'https://data.4tu.nl/ndownloader/files/24070952'),
        ('BPIC16_complaints.csv', 'https://data.4tu.nl/ndownloader/files/24075341'),
        ('BPIC16_questions.csv', 'https://data.4tu.nl/ndownloader/files/24023441'),
        ('BPIC17.xes.gz', 'https://data.4tu.nl/ndownloader/files/24044117'),
        ('BPIC17_offer_log.xes.gz', 'https://data.4tu.nl/ndownloader/files/26563211'),
        ('BPIC18.xes.gz', 'https://data.4tu.nl/ndownloader/files/24025820'),
        ('BPIC19.xes', 'https://data.4tu.nl/ndownloader/files/24072995'),
        ('BPIC20_domestic_declarations.xes.gz', 'https://data.4tu.nl/ndownloader/files/24031811'),
        ('BPIC20_international_declarations.xes.gz', 'https://data.4tu.nl/ndownloader/files/24023492'),
        ('BPIC20_prepaid_travel_cost.xes.gz', 'https://data.4tu.nl/ndownloader/files/24043835'),
        ('BPIC20_permit_log.xes.gz', 'https://data.4tu.nl/ndownloader/files/24075929'),
        ('BPIC20_request_for_payment.xes.gz', 'https://data.4tu.nl/ndownloader/files/24061154'),
    ]

    for file_name, url in tqdm(logs):
        r = requests.get(url, allow_redirects=True)
        if file_name.endswith('.gz'):
            open(str(BPIC_DIR / file_name), 'wb').write(r.content)
        else:
            gzip.open(str(BPIC_DIR / file_name) + '.gz', 'wb').write(r.content)
