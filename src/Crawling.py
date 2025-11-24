from selenium import webdriver   # 웹 브라우저 자동 제어
from selenium.webdriver.common.by import By  # HTML 요소를 찾을 때 기준을 지정
from selenium.webdriver.chrome.options import Options  # 크롬 드라이버 옵션 설정
import pandas as pd
import time
from selenium.webdriver.support.ui import WebDriverWait  # 특정 요소가 나타날 때까지 기다림
from selenium.webdriver.support import expected_conditions as EC  # 기다리는 조건을 정의
import re
import os

def file_save(df):

    # 파일을 저장할 폴더 경로
    folder_path = "./data/raw/review"

    # 해당 폴더가 없으면
    if not os.path.exists(folder_path):
        # 폴더 생성
        os.makedirs(folder_path)

    # csv로 저장
    df.to_csv(f"{folder_path}/musinsa_reviews_{goods_no}.csv", index=False, encoding="utf-8-sig")

def crawling(goods_no, target_count):

    # '6만개 이상', '7만회 이상', '12,543회' 등 다양한 형식을 int로 변환
    def extract_korean_number(text): 
        
        if not text: 
            return None
    
        text = text.replace(",", "")  # 콤마 제거
    
        # '6만' or '7만회 이상' 같은 패턴 처리
        match = re.search(r"(\d+)(\.\d+)?\s*만", text)
        if match:
            num = float(match.group(1))
            return int(num * 10000)
    
        # 일반 숫자만 있는 경우
        digits = re.findall(r"\d+", text)
        if digits:
            return int(''.join(digits))
    
        return None

    # 크롬 드라이버 옵션 설정
    options = Options()
    options.add_argument("--headless")  # 창 안 띄우고 실행

    # driver : 크롬 브라우저 객체
    driver = webdriver.Chrome(options=options)

    # 1. 상품 상세 페이지에서 상품 정보 크롤링

    product_url = f"https://www.musinsa.com/products/{goods_no}"
    driver.get(product_url)
    time.sleep(2)

    # By.XPATH : HTML 구조를 따라 논리적으로 찾기
    # "//span[contains(@class, 'GoodsName')]" 
    # : 페이지 안의 모든 <span> 태그 중에서, class 속성 안에 'GoodsName'이라는 문자열이 포함된 태그를 찾아라

    # 상품명
    try:
        product_name = driver.find_element(By.XPATH, "//span[contains(@class,'GoodsName-sc')]").text.strip()
    except:
        product_name = None

    # 가격
    try:
        price_text = driver.find_element(
            By.XPATH, "//span[contains(@class, 'Price__CalculatedPrice')]"
        ).text.strip()
    except:
        price_text = None

    # 할인율
    try:
        discount_text = driver.find_element(
            By.XPATH, "//span[contains(@class, 'Price__DiscountRate')]"
        ).text.strip()
    except:
        discount_text = None

    # 후기 수
    try:
        review_count_text = driver.find_element(
            By.XPATH, "//span[contains(@class, 'underline') and contains(text(),'후기')]"
        ).text.strip()
    except:
        review_count_text = None

    # 스크롤 필요 항목 - 조회수, 누적 판매량
    for i in range(0, 2500, 400):
        driver.execute_script(f"window.scrollTo(0, {i})")
        time.sleep(0.3)

    # 조회수
    try:
        # <dt> 안의 텍스트 내용 중 '조회'라는 글자가 포함된 태그를 찾아서,
        # 그 태그(dt) 바로 뒤에 나오는 형제 요소 중 <dd> 태그를 선택하겠다!
        
        # normalize-space() : 공백 제거 후 검색
        # arguments[0] : <dt> 요소
        # nextElementSibling : <dt> 바로 뒤에 있는 형제 요소 <dd> 선택
        # innerText : 해당 요소의 텍스트 추출(-> JavaScript에 의해 동적으로 생성된 텍스트)
        dt_view = driver.find_element(By.XPATH, "//dt[contains(normalize-space(), '조회')]")
        view_text = driver.execute_script("return arguments[0].nextElementSibling.innerText;", dt_view)
    except:
        view_text = None
    
    # 누적 판매량
    try:
        dt_sales = driver.find_element(By.XPATH, "//dt[contains(normalize-space(), '누적')]")
        sales_text = driver.execute_script("return arguments[0].nextElementSibling.innerText;", dt_sales)
    except:
        sales_text = None

    view_count = extract_korean_number(view_text)
    sales_count = extract_korean_number(sales_text)
    price = extract_korean_number(price_text)
    discount_pct = extract_korean_number(discount_text)
    review_count = extract_korean_number(review_count_text)

    print("✅ 상품 정보 크롤링 완료")
    print(f"상품명: {product_name}, 가격: {price}, 할인율(%): {discount_pct}")
    print(f"리뷰 수: {review_count}, 조회수: 약 {view_count} 회 이상, 누적판매: 약 {sales_count} 개 이상")

    driver.quit()

    # 2. 리뷰 더보기 페이지에서 리뷰 데이터 크롤링

    options = Options()
    # 창 띄우기
    # options.add_argument("--headless")

    driver = webdriver.Chrome(options=options)

    # 리뷰 페이지 접근 - 최신순 정렬 전
    review_url = f"https://www.musinsa.com/review/goods/{goods_no}?gf=A"
    driver.get(review_url)
    time.sleep(2)

    # 스크롤 - '유용한순' 보일 정도
    driver.execute_script("window.scrollTo(0, 650)")
    time.sleep(0.5)

    # '유용한순' span 클릭
    try:
        useful_span = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), '유용한순')]"))
        )
        driver.execute_script("arguments[0].click();", useful_span)
        time.sleep(1)
        print("✅ 유용한순 클릭 완료")
    except Exception as e:
        print("⚠️ 유용한순 클릭 실패:", e)

    # 드롭다운에서 '최신순' span 클릭
    try:
        recent_span = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), '최신순')]"))
        )
        driver.execute_script("arguments[0].click();", recent_span)
        time.sleep(2)
        print("✅ 최신순 정렬 완료")
    except Exception as e:
        print("⚠️ 최신순 클릭 실패:", e)

    collected = []
    seen_reviews = set()

    while len(collected) < target_count:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        review_boxes = driver.find_elements(
            By.CSS_SELECTOR, "div.review-list-item__Container-sc-15m8pom-0"
        )

        for box in review_boxes:
            try:
                review_text = box.find_element(
                    By.CSS_SELECTOR,
                    "span.text-body_13px_reg.w-full.text-black.font-pretendard"
                ).text.strip()
            except:
                review_text = None

            if not review_text or review_text in seen_reviews:
                continue  # 이미 본 리뷰는 스킵

            try:
                nickname = box.find_element(
                    By.XPATH, ".//span[contains(@class,'UserProfileSection__Nickname')]"
                ).text.strip()
            except:
                nickname = None

            try:
                star_text = box.find_element(
                    By.CSS_SELECTOR,
                    "span.text-body_13px_semi.font-pretendard"
                ).text.strip()
                star = int(star_text) if star_text.isdigit() else None
            except:
                star = None

            collected.append((nickname, review_text, star))
            seen_reviews.add(review_text)

        print(f"누적 리뷰 개수: {len(collected)}")

        # 스크롤 끝 감지
        if len(collected) >= target_count:
            break

    # DataFrame 생성
    df = pd.DataFrame(collected, columns=["nickname", "review", "star"])

    driver.quit()

    df["product_name"] = product_name
    df["price"] = price
    df["discount_pct"] = discount_pct
    df["review_count"] = review_count
    df["view_count"] = view_count
    df["sales_count"] = sales_count
    
    file_save(df)

if __name__ == "__main__":

    # 상품 ID (무신사 상품번호)
    goods_no = "912314"
    # 가져올 리뷰 수
    target_count = 5000

    crawling(goods_no, target_count)