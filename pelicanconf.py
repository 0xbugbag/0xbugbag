from datetime import datetime

AUTHOR = 'beo_hijau'
SITENAME = 'beo_hijau | Welcome'
SITEURL = "http://localhost:8000"
SITETITLE = 'beo_hijau'
SITESUBTITLE = 'Berani Bermimpi'
SITELOGO = SITEURL + '/images/profile.png'
FAVICON = SITEURL + '/favicon.ico'
CUSTOM_CSS = 'static/custom.css'


# Sitemap Settings
SITEMAP = {
        'format': 'xml',
    'priorities': {
        'articles': 0.6,
        'indexes': 0.6,
        'pages': 0.5,
    },
    'changefreqs': {
        'articles': 'monthly',
        'indexes': 'daily',
        'pages': 'monthly',
    }
}

PATH = "content"

STATIC_PATHS = ['images', 'static']

EXTRA_PATH_METADATA = {
    'static/custom.css': {'path': 'static/custom.css'},
    #'extra/robots.txt': {'path': 'robots.txt'},
    'images/favicon.ico': {'path': 'favicon.ico'},
    #'extra/CNAME': {'path': 'CNAME'},
    #'extra/LICENSE': {'path': 'LICENSE'},
    #'extra/README': {'path': 'README'},
}

TIMEZONE = 'Asia/Jakarta'
I18N_TEMPLATES_LANG = 'en'
DEFAULT_LANG = 'en'
OG_LOCALE = 'en_US'
LOCALE = 'en_US'

DATE_FORMATS = {
    'en': '%B %d, %Y',
}

MAIN_MENU = True
MENUITEMS = (
    ('Archives', '/archives.html'),
    ('Categories', '/categories.html'),
    ('Tags', '/tags.html'),)

# Code highlight the Theme
PYGMENTS_STYLE = 'friendly'

ARTICLE_URL = '{date:%Y}/{date:%m}/{slug}/'
ARTICLE_SAVE_AS = '{date:%Y}/{date:%m}/{slug}/' + 'index.html' 

PAGE_URL = '{slug}/'
PAGE_SAVE_AS = PAGE_URL + 'index.html'

ARCHIVES_SAVE_AS = 'archives.html'
YEAR_ARCHIVE_SAVE_AS = '{date:%Y}/index.html'
MONTH_ARCHIVE_SAVE_AS = '{date:%Y}/{date:%m}/index.html'

# Feed generation is usually not desired when developing
FEED_DOMAIN = SITEURL
FEED_ALL_ATOM = 'feeds/all.atom.xml'
CATEGORY_FEED_ATOM = 'feeds/{slug}.atom.xml'
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# HOME_HIDE_TAGS
FEED_USE_SUMMARY = True

# Blogroll
#LINKS = (
#    ("Pelican", "https://getpelican.com/"),
#    ("Python.org", "https://www.python.org/"),
#    ("Jinja2", "https://palletsprojects.com/p/jinja/"),
#    ("You can modify those links in your config file", "#"),
#)

CC_LICENSE = {
    'name': 'Creative Commons Attribution-ShareAlike',
    'version': '4.0',
    'slug': 'by-sa'
}

COPYRIGHT_NAME = '0xp_'
COPYRIGHT_YEAR = datetime.now().year
DEFAULT_PAGINATION = 5

# Uncomment following line if you want document-relative URLs when developing
# RELATIVE_URLS = True

DISABLE_URL_HASH = True

THEME = 'themes/Flex'

#PLUGIN_PATHS = ['./pelican-plugins']
#PLUGINS = ['sitemap', 'post_stats', 'feed_summary']

# Add a link to your social media accounts
SOCIAL = (
    ('github', 'https://github.com/0xbugbag'),
    ('envelope', 'mailto:0xpotchgen.ui@gmail.com'),
    ('linkedin','https://np.linkedin.com/id/hanumaditya'),
    ('twitter','https://twitter.com/sxbugbag'),
)