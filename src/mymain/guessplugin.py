# -*- coding: utf-8-*-
from __future__ import print_function
from __future__ import absolute_import
import logging
import sys
sys.path.append('/home/.dingdang/myplugins/plugincode/genda/mymain/')#为python添加guessgenda绝对路径！！！
from guessgenda import guess

reload(sys)
sys.setdefaultencoding('utf8')

# Standard module stuff
WORDS = ["XINGBIE"]
SLUG = "genda"


def handle(text, mic, profile, wxbot=None):
    """
    Responds to user-input, typically speech text
    Arguments:
        text -- user-input, typically transcribed speech
        mic -- used to interact with the user (for both input and output)
        profile -- contains information related to the user (e.g., phone
                   number)
        wxbot -- wechat bot instance
    """
    logger = logging.getLogger(__name__)
    # get config
    if SLUG not in profile or \
            'age' not in profile[SLUG]:
        mic.say('性别检测插件配置有误，插件使用失败', cache=True)
        return
    age = profile[SLUG]['age']
    try:
        gen = guess()
        age = guess(model_dir='/home/.dingdang/myplugins/plugincode/22801',class_type='age')#使用绝对路径路径
        logger.debug("genda report: ", gen)
        if gen=='M':
            mic.say('帅哥你好！', cache=True)
            print('prediction:',age)
        else:
            mic.say('美女你好！', cache=True)
            print('prediction:',age)
    except Exception, e:
        logger.error(e)


def isValid(text):
    """
        Returns True if the input is related to weather.
        Arguments:
        text -- user-input, typically transcribed speech
    """


    return any(word in text for word in [u"我好看么", u"称赞"])
