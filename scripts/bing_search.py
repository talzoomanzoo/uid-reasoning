import os
import json
import requests
from requests.exceptions import Timeout
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import concurrent
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
from io import BytesIO
import re
import string
from typing import Optional, Tuple
from nltk.tokenize import sent_tokenize


# ----------------------- 自定义请求头 -----------------------
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/58.0.3029.110 Safari/537.36',
    'Referer': 'https://www.google.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}
jina_headers = {
    'Authorization': 'Bearer jina_xxxxx',
    'X-Return-Format': 'markdown',
    # 'X-With-Links-Summary': 'true'
}

# 初始化会话
session = requests.Session()
session.headers.update(headers)



def remove_punctuation(text: str) -> str:
    """移除文本中的标点符号。"""
    return text.translate(str.maketrans("", "", string.punctuation))

def f1_score(true_set: set, pred_set: set) -> float:
    """计算两个词集合之间的 F1 分数。"""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)

def extract_snippet_with_context(full_text: str, snippet: str, context_chars: int = 2500) -> Tuple[bool, str]:
    """
    从完整文本中提取与 snippet 最匹配的句子及其上下文。

    Args:
        full_text (str): 从网页中提取的完整文本。
        snippet (str): 要匹配的片段。
        context_chars (int): 片段前后要包含的字符数。

    Returns:
        Tuple[bool, str]: 第一个元素表示是否成功提取，第二个元素是提取的上下文内容。
    """
    try:
        full_text = full_text[:50000]

        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2

        # sentences = re.split(r'(?<=[.!?]) +', full_text)  # 使用正则表达式分割句子，支持 ., !, ? 结尾
        sentences = sent_tokenize(full_text)  # 使用 nltk 的 sent_tokenize 分割句子

        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            # 如果未找到匹配句子，返回全文的前 context_chars*2 字符
            return False, full_text[:context_chars * 2]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"

def extract_text_from_url(url, use_jina=False, snippet: Optional[str] = None):
    """
    从 URL 中提取文本。如果提供了 snippet，则提取与之相关的上下文。

    Args:
        url (str): 网页或 PDF 的 URL。
        use_jina (bool): 是否使用 Jina 进行提取。
        snippet (Optional[str]): 要查找的片段。

    Returns:
        str: 提取的文本或上下文。
    """
    try:
        if use_jina:
            response = requests.get(f'https://r.jina.ai/{url}', headers=jina_headers).text
            # 去除 URL
            pattern = r"\(https?:.*?\)|\[https?:.*?\]"
            text = re.sub(pattern, "", response).replace('---','-').replace('===','=').replace('   ',' ').replace('   ',' ')
        else:
            response = session.get(url, timeout=20)  # 设置超时时间为20秒
            response.raise_for_status()  # 如果请求失败，抛出 HTTPError
            # 判断返回的内容类型
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' in content_type:
                # 如果是 PDF 文件，提取 PDF 文本
                return extract_pdf_text(url)
            # 尝试使用 lxml 解析，如果不可用则使用 html.parser
            try:
                soup = BeautifulSoup(response.text, 'lxml')
            except Exception:
                print("lxml parser not found or failed, falling back to html.parser")
                soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)

        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            if success:
                return context
            else:
                return text
        else:
            # 如果未提供片段，则直接返回
            return text[:8000]
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.ConnectionError:
        return "Error: Connection error occurred"
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def fetch_page_content(urls, max_workers=4, use_jina=False, snippets: Optional[dict] = None):
    """
    并发地从多个 URL 中获取内容。

    Args:
        urls (list): 要抓取的 URL 列表。
        max_workers (int): 最大并发线程数。
        use_jina (bool): 是否使用 Jina 进行提取。
        snippets (Optional[dict]): 一个字典，将 URL 映射到相应的片段。

    Returns:
        dict: 一个字典，将 URL 映射到提取的内容或上下文。
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用 tqdm 显示进度条
        futures = {
            executor.submit(extract_text_from_url, url, use_jina, snippets.get(url) if snippets else None): url
            for url in urls
        }
        for future in tqdm(concurrent.futures.as_completed(futures), desc="Fetching URLs", total=len(urls)):
            url = futures[future]
            try:
                data = future.result()
                results[url] = data
            except Exception as exc:
                results[url] = f"Error fetching {url}: {exc}"
            time.sleep(0.2)  # 简单的速率限制
    return results


def bing_web_search(query, subscription_key, endpoint, market='en-US', language='en', timeout=20):
    """
    使用 Bing Web Search API 进行搜索，并设置超时时间。

    Args:
        query (str): 搜索查询。
        subscription_key (str): Bing 搜索 API 的订阅密钥。
        endpoint (str): Bing 搜索 API 的终端。
        market (str): 市场，例如 "en-US" 或 "zh-CN"。
        language (str): 返回的语言，例如 "en"。
        timeout (int or float or tuple): 请求超时时间，单位为秒。
                                         可以是一个浮点数表示总的超时时间，
                                         也可以是一个 (connect timeout, read timeout) 元组。

    Returns:
        dict: 搜索结果的 JSON 响应。如果请求超时，则返回 None 或抛出异常。
    """
    headers = {
        "Ocp-Apim-Subscription-Key": subscription_key
    }
    params = {
        "q": query,
        "mkt": market,
        "setLang": language,
        "textDecorations": True,
        "textFormat": "HTML"
    }

    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=timeout)
        response.raise_for_status()  # 如果请求失败，抛出异常
        search_results = response.json()
        return search_results
    except Timeout:
        print(f"请求 Bing Web Search 超时 ({timeout} 秒) for query: {query}")
        return {}  # 或者你可以选择抛出异常
    except requests.exceptions.RequestException as e:
        print(f"请求 Bing Web Search 出错: {e}")
        return {}


def extract_pdf_text(url):
    """
    从 PDF 中提取文本。

    Args:
        url (str): PDF 文件的 URL。

    Returns:
        str: 提取的文本内容或错误信息。
    """
    try:
        response = session.get(url, timeout=20)  # 设置超时时间为20秒
        if response.status_code != 200:
            return f"Error: Unable to retrieve the PDF (status code {response.status_code})"
        
        # 使用 pdfplumber 打开 PDF 文件
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text
        
        # 限制文本长度
        cleaned_text = ' '.join(full_text.split()[:600])
        return cleaned_text
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Error: {str(e)}"

def extract_relevant_info(search_results):
    """
    从 Bing 搜索结果中提取相关信息。

    Args:
        search_results (dict): Bing Web Search API 的 JSON 响应。

    Returns:
        list: 包含提取信息的字典列表。
    """
    useful_info = []
    
    if 'webPages' in search_results and 'value' in search_results['webPages']:
        for id, result in enumerate(search_results['webPages']['value']):
            info = {
                'id': id + 1,  # 增加 id，方便后续操作
                'title': result.get('name', ''),
                'url': result.get('url', ''),
                'site_name': result.get('siteName', ''),
                'date': result.get('datePublished', '').split('T')[0],
                'snippet': result.get('snippet', ''),  # 去除 HTML 标签
                # 将上下文内容添加到信息中
                'context': ''  # 预留字段，将在后续填充
            }
            useful_info.append(info)
    
    return useful_info


# ------------------------------------------------------------

if __name__ == "__main__":
    # 示例用法
    # 定义要搜索的查询
    query = "Structure of dimethyl fumarate"
    
    # Bing搜索API的订阅密钥和终端
    BING_SUBSCRIPTION_KEY = "YOUR_BING_SUBSCRIPTION_KEY"
    if not BING_SUBSCRIPTION_KEY:
        raise ValueError("Please set the BING_SEARCH_V7_SUBSCRIPTION_KEY environment variable.")
    
    bing_endpoint = "https://api.bing.microsoft.com/v7.0/search"
    
    # 执行搜索
    print("Performing Bing Web Search...")
    search_results = bing_web_search(query, BING_SUBSCRIPTION_KEY, bing_endpoint)
    
    print("Extracting relevant information from search results...")
    extracted_info = extract_relevant_info(search_results)

    print("Fetching and extracting context for each snippet...")
    for info in tqdm(extracted_info, desc="Processing Snippets"):
        full_text = extract_text_from_url(info['url'], use_jina=True)  # 获取完整网页文本
        if full_text and not full_text.startswith("Error"):
            success, context = extract_snippet_with_context(full_text, info['snippet'])
            if success:
                info['context'] = context
            else:
                info['context'] = f"Could not extract context. Returning first 8000 chars: {full_text[:8000]}"
        else:
            info['context'] = f"Failed to fetch full text: {full_text}"

    # print("Your Search Query:", query)
    # print("Final extracted information with context:")
    # print(json.dumps(extracted_info, indent=2, ensure_ascii=False))