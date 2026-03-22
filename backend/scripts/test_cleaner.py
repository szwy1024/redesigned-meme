"""Test script for TextCleaner."""

from app.dataset.text_cleaner import TextCleaner, get_default_cleaner


def main() -> None:
    cleaner = get_default_cleaner()

    test_cases = [
        "今天天气真好！https://example.com 😊 #开心",
        "<p>这是一段HTML文本</p> 记得点赞 👍👍👍 @用户1",
        "收到！！！太棒了???? https://t.co/abc123 #Python #机器学习",
        "<div class='test'>HTML &amp; URL</div> 今天学习 😎 #AI",
        "这个商品太差了！！！☠️💔 #失望",
        "学习使我快乐！！！😊😊😊 #学习 #进步",
    ]

    print("=" * 60)
    print("TextCleaner Test Results")
    print("=" * 60)

    for i, text in enumerate(test_cases, 1):
        print(f"\n[Test {i}]")
        print(f"  Original: {text}")
        result = cleaner.clean(text)
        print(f"  Cleaned:  {result.cleaned_text}")
        print(f"  URLs removed: {result.removed_urls}")
        print(f"  HTML tags removed: {result.removed_html_tags}")
        print(f"  Emojis converted: {result.converted_emojis}")
        print(f"  Hashtags: {result.extracted_hashtags}")
        print(f"  Mentions: {result.extracted_mentions}")


if __name__ == "__main__":
    main()
