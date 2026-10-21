"""
Feature Extraction for PhiUSIIL Phishing URL Dataset (50 Features)
Extracts URL lexical, structural, and content-based features.
"""

import os
import re
import math
import pickle
import ipaddress
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import numpy as np

# Load precomputed TLD legitimate probability map
TLD_MAP = {}
tld_map_path = os.path.join(os.path.dirname(__file__), "pickle", "tld_map.pkl")
if os.path.exists(tld_map_path):
    try:
        with open(tld_map_path, "rb") as f:
            TLD_MAP = pickle.load(f)
    except Exception:
        TLD_MAP = {}



class FeatureExtraction:
    def __init__(self, url):
        self.raw_url = str(url).strip()
        self.url = self.raw_url
        if not self.url.startswith(('http://', 'https://')):
            self.url = 'http://' + self.url

        try:
            self.urlparse = urlparse(self.url)
            self.domain = self.urlparse.netloc.lower()
            if ':' in self.domain:
                self.domain = self.domain.split(':')[0]
        except Exception:
            self.urlparse = None
            self.domain = ""

        # Extract TLD
        domain_parts = self.domain.split('.')
        self.tld = domain_parts[-1] if len(domain_parts) > 1 else ""

        # Network fetch with 2s timeout
        self.response = None
        self.soup = None
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
        }
        try:
            self.response = requests.get(self.url, headers=headers, timeout=2.0, allow_redirects=True)
            self.soup = BeautifulSoup(self.response.text, 'html.parser')
        except Exception:
            self.response = None
            self.soup = None

        self.features = self.extract_all()

    def extract_all(self):
        url_str = self.url
        dom_str = self.domain
        url_len = len(url_str)
        dom_len = len(dom_str)

        # 1. URLLength
        url_length = url_len

        # 2. DomainLength
        domain_length = dom_len

        # 3. IsDomainIP
        is_domain_ip = 0
        try:
            ipaddress.ip_address(dom_str)
            is_domain_ip = 1
        except Exception:
            is_domain_ip = 0

        # 4. CharContinuationRate (consecutive alphanumeric continuity)
        clean_dom = re.sub(r'^www\.', '', dom_str)
        dom_parts = clean_dom.split('.')
        main_label = dom_parts[0] if dom_parts else clean_dom
        if re.match(r'^[a-zA-Z]+$', main_label):
            char_cont_rate = 1.0
        else:
            alnum_sequences = re.findall(r'[a-zA-Z0-9]+', main_label)
            longest_seq = max([len(s) for s in alnum_sequences], default=0)
            char_cont_rate = min(1.0, longest_seq / max(len(main_label), 1))

        # 6. TLDLegitimateProb
        tld_prob = TLD_MAP.get(self.tld, 0.05 if is_domain_ip else 0.15)

        # Character counts
        letters = re.findall(r'[a-zA-Z]', url_str)
        no_of_letters = len(letters)
        letter_ratio = no_of_letters / max(url_len, 1)

        digits = re.findall(r'[0-9]', url_str)
        no_of_digits = len(digits)
        digit_ratio = no_of_digits / max(url_len, 1)

        # 6. URLCharProb
        url_char_prob = max(0.02, min(0.08, 0.05 + 0.02 * (no_of_letters / max(url_len, 1)) - 0.02 * (no_of_digits / max(url_len, 1))))

        # 8. TLDLength
        tld_length = len(self.tld)

        # 9. NoOfSubDomain
        clean_dom = dom_str.replace('www.', '')
        dot_count = clean_dom.count('.')
        no_of_subdomain = max(0, dot_count)

        # 10. HasObfuscation
        has_obfuscation = 1 if ('%' in url_str or '@' in url_str) else 0

        # 11. NoOfObfuscatedChar
        no_of_obfuscated_char = len(re.findall(r'%[0-9a-fA-F]{2}', url_str))

        # 12. ObfuscationRatio
        obfuscation_ratio = no_of_obfuscated_char / max(url_len, 1)

        # 13. NoOfLettersInURL (already computed)
        # 14. LetterRatioInURL (already computed)
        # 15. NoOfDegitsInURL (already computed)
        # 16. DegitRatioInURL (already computed)

        # 17. NoOfEqualsInURL
        no_of_equals = url_str.count('=')

        # 18. NoOfQMarkInURL
        no_of_qmark = url_str.count('?')

        # 19. NoOfAmpersandInURL
        no_of_ampersand = url_str.count('&')

        # 20. NoOfOtherSpecialCharsInURL
        special_chars = re.findall(r'[^a-zA-Z0-9\.\/\:\?\=\&\-]', url_str)
        no_of_other_special = len(special_chars)

        # 21. SpacialCharRatioInURL
        all_special = re.findall(r'[^a-zA-Z0-9]', url_str)
        special_char_ratio = len(all_special) / max(url_len, 1)

        # 22. IsHTTPS
        is_https = 1 if (self.urlparse and self.urlparse.scheme == 'https') else 0

        # --- HTML / Content Features (computed if page returned, else safe default 0) ---
        line_of_code = 0
        largest_line_len = 0
        has_title = 0
        dom_title_match = 0.0
        url_title_match = 0.0
        has_favicon = 0
        robots = 0
        is_responsive = 0
        no_redirect = 0
        no_self_redirect = 0
        has_desc = 0
        no_popup = 0
        no_iframe = 0
        has_ext_form = 0
        has_social = 0
        has_submit_btn = 0
        has_hidden_fields = 0
        has_pwd_field = 0
        has_copyright = 0
        no_img = 0
        no_css = 0
        no_js = 0
        no_self_ref = 0
        no_empty_ref = 0
        no_ext_ref = 0

        # Sensitive Keywords
        lower_url = url_str.lower()
        bank_kw = 1 if any(k in lower_url for k in ['bank', 'chase', 'wellsfargo', 'citi', 'barclays', 'hsbc']) else 0
        pay_kw = 1 if any(k in lower_url for k in ['pay', 'paypal', 'applepay', 'checkout', 'billing', 'invoice']) else 0
        crypto_kw = 1 if any(k in lower_url for k in ['crypto', 'bitcoin', 'eth', 'binance', 'coinbase', 'wallet', 'airdrop']) else 0

        if self.response and self.soup:
            text = self.response.text
            lines = text.splitlines()
            line_of_code = max(len(lines), len(text) // 70)
            largest_line_len = max([len(l) for l in lines], default=0)

            # Title
            title_tag = self.soup.find('title')
            if title_tag and title_tag.string:
                has_title = 1
                t_str = title_tag.string.lower()
                dom_words = [w for w in re.split(r'\W+', dom_str) if len(w) > 2]
                match_count = sum(1 for w in dom_words if w in t_str)
                dom_title_match = match_count / max(len(dom_words), 1)
                url_title_match = 1.0 if any(w in t_str for w in dom_words) else 0.0

            # Favicon
            for link in self.soup.find_all('link', href=True):
                rel = link.get('rel', [])
                if any('icon' in str(r).lower() for r in rel):
                    has_favicon = 1
                    break

            # Responsive / Mobile Viewport
            if self.soup.find('meta', attrs={'name': re.compile(r'viewport', re.I)}):
                is_responsive = 1

            # Description
            if self.soup.find('meta', attrs={'name': re.compile(r'description', re.I)}):
                has_desc = 1

            # Redirects
            no_redirect = len(self.response.history)

            # Popups & iFrames
            no_popup = len(re.findall(r'window\.open|alert\(|prompt\(', text))
            no_iframe = len(self.soup.find_all(['iframe', 'frame']))

            # Forms
            forms = self.soup.find_all('form')
            for form in forms:
                action = form.get('action', '')
                if action.startswith('http') and dom_str not in action:
                    has_ext_form = 1
                if form.find(['button', 'input'], attrs={'type': 'submit'}):
                    has_submit_btn = 1

            if self.soup.find('input', attrs={'type': 'hidden'}):
                has_hidden_fields = 1
            if self.soup.find('input', attrs={'type': 'password'}):
                has_pwd_field = 1

            # Social Networks
            social_domains = ['facebook.com', 'twitter.com', 'x.com', 'instagram.com', 'linkedin.com', 'youtube.com']
            for a in self.soup.find_all('a', href=True):
                href = a['href'].lower()
                if any(sd in href for sd in social_domains):
                    has_social = 1
                    break

            # Copyright
            if re.search(r'copyright|&copy;|©', text, re.I):
                has_copyright = 1

            # Tags counts
            no_img = len(self.soup.find_all('img'))
            no_css = len(self.soup.find_all(['link'], attrs={'rel': re.compile(r'stylesheet', re.I)}))
            no_js = len(self.soup.find_all('script'))

            # References
            for a in self.soup.find_all('a', href=True):
                href = a.get('href', '').strip()
                if href in ['', '#', 'javascript:void(0)', 'javascript:;']:
                    no_empty_ref += 1
                elif href.startswith('/') or dom_str in href:
                    no_self_ref += 1
                elif href.startswith('http'):
                    no_ext_ref += 1

        # Construct dictionary matching exact 49 genuine features
        data = {
            'URLLength': url_length,
            'DomainLength': domain_length,
            'IsDomainIP': is_domain_ip,
            'CharContinuationRate': char_cont_rate,
            'TLDLegitimateProb': tld_prob,
            'URLCharProb': url_char_prob,
            'TLDLength': tld_length,
            'NoOfSubDomain': no_of_subdomain,
            'HasObfuscation': has_obfuscation,
            'NoOfObfuscatedChar': no_of_obfuscated_char,
            'ObfuscationRatio': obfuscation_ratio,
            'NoOfLettersInURL': no_of_letters,
            'LetterRatioInURL': letter_ratio,
            'NoOfDegitsInURL': no_of_digits,
            'DegitRatioInURL': digit_ratio,
            'NoOfEqualsInURL': no_of_equals,
            'NoOfQMarkInURL': no_of_qmark,
            'NoOfAmpersandInURL': no_of_ampersand,
            'NoOfOtherSpecialCharsInURL': no_of_other_special,
            'SpacialCharRatioInURL': special_char_ratio,
            'IsHTTPS': is_https,
            'LineOfCode': line_of_code,
            'LargestLineLength': largest_line_len,
            'HasTitle': has_title,
            'DomainTitleMatchScore': dom_title_match,
            'URLTitleMatchScore': url_title_match,
            'HasFavicon': has_favicon,
            'Robots': robots,
            'IsResponsive': is_responsive,
            'NoOfURLRedirect': no_redirect,
            'NoOfSelfRedirect': no_self_redirect,
            'HasDescription': has_desc,
            'NoOfPopup': no_popup,
            'NoOfiFrame': no_iframe,
            'HasExternalFormSubmit': has_ext_form,
            'HasSocialNet': has_social,
            'HasSubmitButton': has_submit_btn,
            'HasHiddenFields': has_hidden_fields,
            'HasPasswordField': has_pwd_field,
            'Bank': bank_kw,
            'Pay': pay_kw,
            'Crypto': crypto_kw,
            'HasCopyrightInfo': has_copyright,
            'NoOfImage': no_img,
            'NoOfCSS': no_css,
            'NoOfJS': no_js,
            'NoOfSelfRef': no_self_ref,
            'NoOfEmptyRef': no_empty_ref,
            'NoOfExternalRef': no_ext_ref
        }
        return data

    def getFeaturesDataFrame(self):
        """Return features as a 2D numpy array (row vector) matching trained model expectations."""
        return np.array([list(self.features.values())], dtype=float)

    def getFeaturesList(self):
        return list(self.features.values())

    def getFeaturesDict(self):
        """Human-readable cards for web UI inspection."""
        items = []
        for k, v in self.features.items():
            status = "Safe"
            if k in ['IsDomainIP', 'HasObfuscation', 'HasExternalFormSubmit'] and v == 1:
                status = "High Risk"
            elif k in ['IsHTTPS', 'HasCopyrightInfo', 'HasSocialNet', 'IsResponsive'] and v == 0:
                status = "Warning"

            items.append({
                'name': k,
                'value': v,
                'status': status
            })
        return items
