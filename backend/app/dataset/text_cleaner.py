"""Social media text preprocessing utilities.

This module provides comprehensive text cleaning for social media content,
handling HTML tags, URLs, emojis, hashtags, and various noise patterns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern


# ============== Emoji Mapping ==============

EMOJI_TO_TEXT: dict[str, str] = {
    # Positive emotions
    "😀": "[笑脸]",
    "😃": "[笑脸]",
    "😄": "[笑脸]",
    "😁": "[微笑]",
    "😆": "[大笑]",
    "😅": "[苦笑]",
    "🤣": "[笑哭]",
    "😂": "[笑哭]",
    "🙂": "[微笑]",
    "😊": "[微笑]",
    "😇": "[天使]",
    "🥰": "[爱慕]",
    "😍": "[爱慕]",
    "🤩": "[崇拜]",
    "😘": "[飞吻]",
    "😗": "[亲亲]",
    "😚": "[亲亲]",
    "😋": "[馋嘴]",
    "😛": "[调皮]",
    "😜": "[调皮]",
    "🤪": "[调皮]",
    "😝": "[调皮]",
    "🤑": "[拜金]",
    "🤗": "[拥抱]",
    "🤭": "[害羞]",
    "🤫": "[捂嘴]",
    "🤔": "[思考]",
    "🤐": "[闭嘴]",
    "🤨": "[质疑]",
    "😐": "[面无表情]",
    "😑": "[面无表情]",
    "😶": "[面无表情]",
    "😏": "[得意]",
    "😒": "[无语]",
    "😌": "[放松]",
    "😔": "[忧郁]",
    "😪": "[困倦]",
    "🤤": "[流口水]",
    "😴": "[睡觉]",
    "🤧": "[感冒]",
    "🥺": "[可怜]",
    "🥹": "[感动]",
    "😎": "[酷]",
    "🤓": "[书呆子]",
    "🧐": "[观察]",
    # Negative emotions
    "😞": "[失望]",
    "😓": "[尴尬]",
    "😩": "[疲惫]",
    "😫": "[疲惫]",
    "🥱": "[无聊]",
    "😤": "[生气]",
    "😡": "[愤怒]",
    "😠": "[愤怒]",
    "🤬": "[脏话]",
    "😈": "[恶魔]",
    "👿": "[恶魔]",
    "💀": "[骷髅]",
    "☠️": "[骷髅]",
    "💩": "[屎]",
    "🤡": "[小丑]",
    "👹": "[妖怪]",
    "👺": "[妖怪]",
    "👻": "[幽灵]",
    "👽": "[外星人]",
    "🤖": "[机器人]",
    # Hearts & love
    "❤️": "[红心]",
    "🧡": "[橙心]",
    "💛": "[黄心]",
    "💚": "[绿心]",
    "💙": "[蓝心]",
    "💜": "[紫心]",
    "🖤": "[黑心]",
    "🤍": "[白心]",
    "💔": "[心碎]",
    "❣️": "[爱心]",
    "💕": "[双心]",
    "💞": "[双心]",
    "💓": "[心跳]",
    "💗": "[心跳]",
    "💖": "[心跳]",
    "💘": "[丘比特]",
    "💝": "[礼物]",
    # Hands & gestures
    "👍": "[点赞]",
    "👎": "[点踩]",
    "👏": "[鼓掌]",
    "🙌": "[欢呼]",
    "🤝": "[握手]",
    "🙏": "[祈祷]",
    "✌️": "[胜利]",
    "🤞": "[好运]",
    "✋": "[掌心]",
    "🤚": "[手背]",
    "👋": "[挥手]",
    "🤙": "[call]",
    "💪": "[肌肉]",
    # Objects & symbols
    "🔥": "[火焰]",
    "⭐": "[星星]",
    "✨": "[闪光]",
    "💥": "[爆炸]",
    "💫": "[旋转]",
    "🎉": "[庆祝]",
    "🎊": "[庆祝]",
    "🎈": "[气球]",
    "🎁": "[礼物]",
    "🏆": "[奖杯]",
    "🥇": "[金牌]",
    "🥈": "[银牌]",
    "🥉": "[铜牌]",
    "⚽": "[足球]",
    "🏀": "[篮球]",
    "🎯": "[靶心]",
    "🎱": "[台球]",
    "🎮": "[游戏]",
    "🎲": "[骰子]",
    "🃏": "[扑克]",
    "🀄": "[麻将]",
    "🎵": "[音符]",
    "🎶": "[音符]",
    "🎤": "[麦克风]",
    "🎧": "[耳机]",
    "📱": "[手机]",
    "💻": "[电脑]",
    "⌨️": "[键盘]",
    "🖥️": "[显示器]",
    "🖱️": "[鼠标]",
    "📷": "[相机]",
    "📸": "[相机]",
    "📹": "[录像]",
    "🎥": "[摄像机]",
    "📺": "[电视]",
    "📻": "[收音机]",
    "🕐": "[时钟]",
    "⌚": "[手表]",
    "📚": "[书籍]",
    "📖": "[阅读]",
    "✏️": "[铅笔]",
    "📝": "[笔记]",
    "📌": "[图钉]",
    "📍": "[地点]",
    "🚩": "[旗帜]",
    "💰": "[钱袋]",
    "💵": "[钞票]",
    "💳": "[银行卡]",
    "💎": "[钻石]",
    "🔑": "[钥匙]",
    "🔒": "[锁]",
    "🔓": "[开锁]",
    "🔗": "[链接]",
    "📦": "[包裹]",
    "🏠": "[房子]",
    "🏢": "[建筑]",
    "🏪": "[商店]",
    "🏫": "[学校]",
    "🏥": "[医院]",
    "🏦": "[银行]",
    "🏨": "[酒店]",
    "🏩": "[旅馆]",
    "🏬": "[商场]",
    "🏭": "[工厂]",
    "🏯": "[日本城堡]",
    "🏰": "[城堡]",
    "🗼": "[东京塔]",
    "🗽": "[自由女神]",
    "⛪": "[教堂]",
    "🕌": "[清真寺]",
    "🛕": "[寺庙]",
    "⛩️": "[神社]",
    "🗻": "[富士山]",
    "🌋": "[火山]",
    "🏔️": "[山峰]",
    "🏕️": "[露营]",
    "🏖️": "[海滩]",
    "🏜️": "[沙漠]",
    "🏝️": "[岛屿]",
    "🏞️": "[公园]",
    "🌅": "[日出]",
    "🌄": "[日出]",
    "🌃": "[夜景]",
    "🌉": "[桥]",
    "🌌": "[星空]",
    "🎑": "[月祭]",
    "🌙": "[月亮]",
    "🌛": "[月亮]",
    "🌜": "[月亮]",
    "🌚": "[月亮]",
    "🌝": "[月亮]",
    "🌞": "[太阳]",
    "☀️": "[太阳]",
    "🌤️": "[晴朗]",
    "⛅": "[多云]",
    "🌥️": "[多云]",
    "🌦️": "[阵雨]",
    "🌧️": "[雨天]",
    "⛈️": "[雷雨]",
    "🌩️": "[雷电]",
    "🌨️": "[雪天]",
    "❄️": "[雪花]",
    "☃️": "[雪人]",
    "⛄": "[雪人]",
    "🌈": "[彩虹]",
    "🌂": "[雨伞]",
    "☂️": "[雨伞]",
    "☔": "[雨伞]",
    "⚡": "[闪电]",
    "💧": "[水滴]",
    "🌊": "[海浪]",
    "🚗": "[汽车]",
    "🚕": "[出租车]",
    "🚙": "[SUV]",
    "🚌": "[公交车]",
    "🚎": "[无轨电车]",
    "🏎️": "[赛车]",
    "🚓": "[警车]",
    "🚑": "[救护车]",
    "🚒": "[消防车]",
    "🚐": "[货车]",
    "🛻": "[皮卡]",
    "🚚": "[卡车]",
    "🚛": "[集装箱车]",
    "🚜": "[拖拉机]",
    "🚲": "[自行车]",
    "🛵": "[摩托车]",
    "🏍️": "[摩托车]",
    "🛺": "[三轮车]",
    "🚏": "[公交站]",
    "🛤️": "[轨道]",
    "🛣️": "[高速公路]",
    "🚢": "[船]",
    "⛵": "[帆船]",
    "🚤": "[快艇]",
    "🛳️": "[客轮]",
    "⛴️": "[渡轮]",
    "🚁": "[直升机]",
    "🛩️": "[飞机]",
    "✈️": "[飞机]",
    "🛫": "[起飞]",
    "🛬": "[降落]",
    "🚀": "[火箭]",
    "🛰️": "[卫星]",
    "💺": "[座位]",
    "🏗️": "[建筑中]",
    "🌉": "[桥]",
    "🚢": "[轮船]",
    # Food & drinks
    "🍎": "[苹果]",
    "🍐": "[梨]",
    "🍊": "[橙子]",
    "🍋": "[柠檬]",
    "🍌": "[香蕉]",
    "🍉": "[西瓜]",
    "🍇": "[葡萄]",
    "🍓": "[草莓]",
    "🫐": "[蓝莓]",
    "🍈": "[哈密瓜]",
    "🍒": "[樱桃]",
    "🍑": "[桃子]",
    "🥭": "[芒果]",
    "🍍": "[菠萝]",
    "🥥": "[椰子]",
    "🥝": "[猕猴桃]",
    "🍅": "[番茄]",
    "🍆": "[茄子]",
    "🥑": "[牛油果]",
    "🥦": "[西兰花]",
    "🥬": "[蔬菜]",
    "🌽": "[玉米]",
    "🥕": "[胡萝卜]",
    "🧄": "[大蒜]",
    "🧅": "[洋葱]",
    "🥔": "[土豆]",
    "🍠": "[烤红薯]",
    "🥐": "[牛角面包]",
    "🥯": "[贝果]",
    "🍞": "[面包]",
    "🥖": "[法棍]",
    "🥨": "[椒盐卷饼]",
    "🧀": "[奶酪]",
    "🥚": "[鸡蛋]",
    "🍳": "[煎蛋]",
    "🧈": "[黄油]",
    "🥞": "[煎饼]",
    "🧇": "[华夫饼]",
    "🥓": "[培根]",
    "🥩": "[牛排]",
    "🍗": "[鸡腿]",
    "🍖": "[排骨]",
    "🌭": "[热狗]",
    "🍔": "[汉堡]",
    "🍟": "[薯条]",
    "🍕": "[披萨]",
    "🥪": "[三明治]",
    "🥙": "[沙拉卷]",
    "🧆": "[沙拉三明治]",
    "🌮": "[墨西哥卷饼]",
    "🌯": "[墨西哥煎饼]",
    "🥗": "[沙拉]",
    "🥘": "[浅锅]",
    "🫕": "[火锅]",
    "🍝": "[意大利面]",
    "🍜": "[拉面]",
    "🍲": "[咖喱饭]",
    "🍛": "[咖喱饭]",
    "🍣": "[寿司]",
    "🍱": "[便当]",
    "🥟": "[饺子]",
    "🥠": "[幸运饼干]",
    "🥮": "[月饼]",
    "🍢": "[关东煮]",
    "🍡": "[团子]",
    "🍦": "[冰淇淋]",
    "🍧": "[刨冰]",
    "🍨": "[冰淇淋]",
    "🍩": "[甜甜圈]",
    "🍪": "[饼干]",
    "🎂": "[蛋糕]",
    "🍰": "[蛋糕]",
    "🧁": "[纸杯蛋糕]",
    "🥧": "[派]",
    "🍫": "[巧克力]",
    "🍬": "[糖果]",
    "🍭": "[棒棒糖]",
    "🍿": "[爆米花]",
    "🍿": "[爆米花]",
    "🍺": "[啤酒]",
    "🍻": "[啤酒杯]",
    "🥂": "[香槟]",
    "🍷": "[葡萄酒]",
    "🥃": "[威士忌]",
    "🍸": "[鸡尾酒]",
    "🍹": "[果汁酒]",
    "🧃": "[纸盒饮料]",
    "🥤": "[饮料]",
    "🧋": "[珍珠奶茶]",
    "🧃": "[吸管杯]",
    "☕": "[咖啡]",
    "🍵": "[茶]",
    "🧃": "[豆奶]",
    "🧋": "[奶茶]",
    # Animals
    "🐶": "[狗]",
    "🐱": "[猫]",
    "🐭": "[老鼠]",
    "🐹": "[仓鼠]",
    "🐰": "[兔子]",
    "🦊": "[狐狸]",
    "🐻": "[熊]",
    "🐼": "[熊猫]",
    "🐨": "[考拉]",
    "🐯": "[老虎]",
    "🦁": "[狮子]",
    "🐮": "[牛]",
    "🐷": "[猪]",
    "🐸": "[青蛙]",
    "🐵": "[猴子]",
    "🙈": "[无视线猴]",
    "🙉": "[无耳猴]",
    "🙊": "[无口猴]",
    "🐒": "[猴子]",
    "🐔": "[鸡]",
    "🐧": "[企鹅]",
    "🐦": "[鸟]",
    "🐤": "[小鸡]",
    "🐣": "[孵化鸡]",
    "🐥": "[小鸡]",
    "🦆": "[鸭子]",
    "🦅": "[鹰]",
    "🦉": "[猫头鹰]",
    "🦇": "[蝙蝠]",
    "🐺": "[狼]",
    "🐗": "[野猪]",
    "🐴": "[马]",
    "🦄": "[独角兽]",
    "🐝": "[蜜蜂]",
    "🐛": "[毛毛虫]",
    "🦋": "[蝴蝶]",
    "🐌": "[蜗牛]",
    "🐞": "[瓢虫]",
    "🐜": "[蚂蚁]",
    "🦟": "[蚊子]",
    "🦗": "[蟋蟀]",
    "🕷️": "[蜘蛛]",
    "🦂": "[蝎子]",
    "🐢": "[乌龟]",
    "🐍": "[蛇]",
    "🦎": "[蜥蜴]",
    "🦖": "[霸王龙]",
    "🦕": "[腕龙]",
    "🐙": "[章鱼]",
    "🦑": "[鱿鱼]",
    "🦐": "[虾]",
    "🦞": "[龙虾]",
    "🦀": "[螃蟹]",
    "🐡": "[河豚]",
    "🐠": "[热带鱼]",
    "🐟": "[鱼]",
    "🐬": "[海豚]",
    "🐳": "[蓝鲸]",
    "🐋": "[鲸鱼]",
    "🦈": "[鲨鱼]",
    "🐊": "[鳄鱼]",
    "🐅": "[虎]",
    "🐆": "[豹]",
    "🦓": "[斑马]",
    "🦍": "[大猩猩]",
    "🦧": "[黑猩猩]",
    "🐘": "[大象]",
    "🦛": "[河马]",
    "🦏": "[犀牛]",
    "🐪": "[骆驼]",
    "🐫": "[双峰骆驼]",
    "🦒": "[长颈鹿]",
    "🦘": "[袋鼠]",
    "🐃": "[水牛]",
    "🐂": "[牛]",
    "🐄": "[奶牛]",
    "🐎": "[马]",
    "🐖": "[猪]",
    "🐏": "[公羊]",
    "🐑": "[绵羊]",
    "🦙": "[羊驼]",
    "🐐": "[山羊]",
    "🦌": "[鹿]",
    "🐕": "[狗]",
    "🐩": "[贵宾犬]",
    "🦮": "[导盲犬]",
    "🐕‍🦺": "[服务犬]",
    "🐈": "[猫]",
    "🐓": "[公鸡]",
    "🦃": "[火鸡]",
    "🦚": "[孔雀]",
    "🦜": "[鹦鹉]",
    "🦢": "[天鹅]",
    "🦩": "[火烈鸟]",
    "🦦": "[水獭]",
    "🦥": "[树懒]",
    "🐁": "[老鼠]",
    "🐀": "[大鼠]",
    "🐿️": "[松鼠]",
    "🦔": "[刺猬]",
}


@dataclass
class CleaningResult:
    """Result of text cleaning operation."""

    cleaned_text: str
    removed_urls: int
    removed_html_tags: int
    converted_emojis: int
    extracted_hashtags: list[str]
    extracted_mentions: list[str]


class TextCleaner:
    """Social media text cleaner with comprehensive cleaning capabilities.

    This class provides utilities for cleaning social media text by removing
    HTML tags, URLs, special characters, and converting emojis to text descriptions.

    Example:
        >>> cleaner = TextCleaner()
        >>> result = cleaner.clean("Check this out! https://example.com 😊 #python")
        >>> print(result.cleaned_text)
        "Check this out! [微笑] #python"
    """

    # Compiled regex patterns for performance
    _HTML_TAG_PATTERN: Pattern[str] = re.compile(r"<[^>]+>")
    _URL_PATTERN: Pattern[str] = re.compile(
        r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
        r"|(?:www\.)?[a-zA-Z0-9][-\w]*(?:\.[a-zA-Z]{2,})+(?:/[^\s]*)?"
    )
    _MENTION_PATTERN: Pattern[str] = re.compile(r"@[\w]+")
    _HASHTAG_PATTERN: Pattern[str] = re.compile(r"#[\w]+")
    _MULTIPLE_EXCLAMATION: Pattern[str] = re.compile(r"!{2,}")
    _MULTIPLE_QUESTION: Pattern[str] = re.compile(r"\?{2,}")
    _MULTIPLE_DIGIT: Pattern[str] = re.compile(r"\d{4,}")
    _SPECIAL_HTML_ENTITIES: Pattern[str] = re.compile(
        r"&[a-zA-Z]+;|&#[0-9]+;|&#x[0-9a-fA-F]+;"
    )
    _GARBLED_CHARS: Pattern[str] = re.compile(
        r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]"
    )
    _REPEATED_PUNCTUATION: Pattern[str] = re.compile(r"([.。,，;；:：!！?？])\1{2,}")
    _EXTRA_WHITESPACE: Pattern[str] = re.compile(r"\s+")

    def __init__(
        self,
        convert_emojis: bool = True,
        extract_hashtags: bool = True,
        extract_mentions: bool = True,
        remove_urls: bool = True,
        remove_html: bool = True,
        normalize_whitespace: bool = True,
        keep_emojis_as_token: bool = True,
    ) -> None:
        """Initialize the TextCleaner.

        Args:
            convert_emojis: Whether to convert emojis to text tokens.
            extract_hashtags: Whether to extract hashtags separately.
            extract_mentions: Whether to extract mentions separately.
            remove_urls: Whether to remove URLs from text.
            remove_html: Whether to remove HTML tags.
            normalize_whitespace: Whether to normalize whitespace.
            keep_emojis_as_token: Keep emoji tokens like [微笑] instead of removing.
        """
        self.convert_emojis = convert_emojis
        self.extract_hashtags = extract_hashtags
        self.extract_mentions = extract_mentions
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.normalize_whitespace = normalize_whitespace
        self.keep_emojis_as_token = keep_emojis_as_token

    def clean(self, text: str) -> CleaningResult:
        """Clean a single text sample.

        Args:
            text: Raw social media text to clean.

        Returns:
            CleaningResult containing cleaned text and statistics.
        """
        if not text:
            return CleaningResult(
                cleaned_text="",
                removed_urls=0,
                removed_html_tags=0,
                converted_emojis=0,
                extracted_hashtags=[],
                extracted_mentions=[],
            )

        original_text = text
        removed_urls = 0
        removed_html_tags = 0
        converted_emojis = 0
        extracted_hashtags: list[str] = []
        extracted_mentions: list[str] = []

        # Step 1: Remove HTML tags
        if self.remove_html:
            html_matches = self._HTML_TAG_PATTERN.findall(text)
            removed_html_tags = len(html_matches)
            text = self._HTML_TAG_PATTERN.sub("", text)
            text = self._SPECIAL_HTML_ENTITIES.sub("", text)

        # Step 2: Remove URLs
        if self.remove_urls:
            url_matches = self._URL_PATTERN.findall(original_text)
            removed_urls = len(url_matches)
            text = self._URL_PATTERN.sub("", text)

        # Step 3: Extract mentions and hashtags
        if self.extract_mentions:
            extracted_mentions = self._MENTION_PATTERN.findall(text)
        if self.extract_hashtags:
            extracted_hashtags = self._HASHTAG_PATTERN.findall(text)

        # Step 4: Convert emojis to text
        if self.convert_emojis:
            for emoji, description in EMOJI_TO_TEXT.items():
                if emoji in text:
                    converted_emojis += text.count(emoji)
                    text = text.replace(emoji, description)

        # Step 5: Remove garbled characters
        text = self._GARBLED_CHARS.sub("", text)

        # Step 6: Normalize repeated punctuation
        text = self._REPEATED_PUNCTUATION.sub(r"\1", text)
        text = self._MULTIPLE_EXCLAMATION.sub("!", text)
        text = self._MULTIPLE_QUESTION.sub("?", text)

        # Step 7: Normalize numbers (optional - replace long numbers)
        text = self._MULTIPLE_DIGIT.sub("NUM", text)

        # Step 8: Remove remaining special characters but keep Chinese
        # Keep: Chinese characters, basic Latin letters, numbers, basic punctuation
        text = self._EXTRA_WHITESPACE.sub(" ", text)

        # Step 9: Trim whitespace
        if self.normalize_whitespace:
            text = text.strip()

        return CleaningResult(
            cleaned_text=text,
            removed_urls=removed_urls,
            removed_html_tags=removed_html_tags,
            converted_emojis=converted_emojis,
            extracted_hashtags=extracted_hashtags,
            extracted_mentions=extracted_mentions,
        )

    def clean_batch(self, texts: list[str]) -> list[CleaningResult]:
        """Clean a batch of text samples.

        Args:
            texts: List of raw social media texts.

        Returns:
            List of CleaningResult objects.
        """
        return [self.clean(text) for text in texts]

    def remove_emoji(self, text: str) -> str:
        """Remove all emojis from text.

        Args:
            text: Text potentially containing emojis.

        Returns:
            Text with all emojis removed.
        """
        for emoji in EMOJI_TO_TEXT:
            text = text.replace(emoji, "")
        return text

    def has_emoji(self, text: str) -> bool:
        """Check if text contains any emojis.

        Args:
            text: Text to check.

        Returns:
            True if text contains any emoji.
        """
        return any(emoji in text for emoji in EMOJI_TO_TEXT)


def get_default_cleaner() -> TextCleaner:
    """Get a TextCleaner with default settings.

    Returns:
        TextCleaner instance with default configuration.
    """
    return TextCleaner(
        convert_emojis=True,
        extract_hashtags=True,
        extract_mentions=True,
        remove_urls=True,
        remove_html=True,
        normalize_whitespace=True,
        keep_emojis_as_token=True,
    )
