import pandas as pd

# CSV 파일 읽기
# fish_population = pd.read_csv('../data/fish_population.csv')
# water_quality = pd.read_csv('../data/water_quality.csv')
fish_population = pd.read_csv('/Users/username/project/data/fish_population.csv')
water_quality = pd.read_csv('/Users/username/project/data/water_quality.csv')

# 데이터 프레임 확인
print(fish_population.head())
print(water_quality.head())

# 주요 환경 문제 식별
# 문제 1: 특정 날짜에 특정 사이트에서 물고기 개체수 감소
fish_population['Date'] = pd.to_datetime(fish_population['Date'])
fish_population_grouped = fish_population.groupby(['Site', 'Date']).sum().reset_index()

# 문제 2: 특정 종의 감소
species_count = fish_population.groupby(['Species', 'Date']).sum().reset_index()

# 수질 데이터와 물고기 개체수 비교
merged_data = pd.merge(fish_population_grouped, water_quality, on=['Site', 'Date'])

# 데이터 분석
print(merged_data)

# 예시 분석 결과 출력
for site in merged_data['Site'].unique():
    site_data = merged_data[merged_data['Site'] == site]
    print(f"Site: {site}")
    print(site_data[['Date', 'Count', 'Average', 'Size (cm)', 'Temperature', 'pH', 'Dissolved Oxygen']])

# 데이터 분석을 통한 문제 해결 방안 제안
# 예: 특정 수질 지표가 물고기 개체수에 미치는 영향을 분석하여, 수질 개선 방안 제시