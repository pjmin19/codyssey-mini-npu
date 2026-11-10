import json
import math
import re
from pathlib import Path
from time import perf_counter_ns


DATA_PATH = Path(__file__).with_name("data.json")
SIZES = (3, 5, 13, 25)
BATCH_SIZES = SIZES[1:]
EPSILON = 1e-9
REPEATS = 1000
LABELS = {"+": "Cross", "cross": "Cross", "x": "X"}


def normalize_label(label):
    if isinstance(label, str):
        normalized = LABELS.get(label.strip().lower())
        if normalized:
            return normalized
    raise ValueError(f"알 수 없는 라벨 {label!r}. Cross(+) 또는 X(x)를 사용하세요.")


def finite_number(value):
    try:
        number = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{value!r}을 유한한 숫자로 읽을 수 없습니다.") from error
    if not math.isfinite(number):
        raise ValueError(f"{value!r}은 유한한 숫자가 아닙니다.")
    return number


def validate_matrix(matrix, size):
    if not isinstance(matrix, list) or len(matrix) != size:
        raise ValueError(f"{size}×{size} 배열에는 행이 {size}개 있어야 합니다.")
    result = []
    for row_number, row in enumerate(matrix, 1):
        if not isinstance(row, list) or len(row) != size:
            raise ValueError(f"{row_number}행의 열이 {size}개여야 합니다.")
        if any(type(value) not in (int, float) for value in row):
            raise ValueError(f"{row_number}행에는 JSON 숫자만 넣을 수 있습니다.")
        result.append([finite_number(value) for value in row])
    return result


def mac(pattern, weights):
    score = 0.0
    for row in range(len(pattern)):
        pattern_row = pattern[row]
        weights_row = weights[row]
        for column in range(len(pattern_row)):
            score += pattern_row[column] * weights_row[column]
    return score


def flatten(matrix):
    return [value for row in matrix for value in row]


def mac_flat(pattern, weights):
    score = 0.0
    for index in range(len(pattern)):
        score += pattern[index] * weights[index]
    return score


def winner(first, second, labels):
    if not math.isfinite(first) or not math.isfinite(second):
        raise ValueError("MAC 점수가 수치 범위를 넘었습니다.")
    if abs(first - second) < EPSILON:
        return "UNDECIDED"
    return labels[0] if first > second else labels[1]


def pattern_for(size, label):
    middle = {size // 2, (size - 1) // 2}
    if label == "Cross":
        return [[int(row in middle or col in middle) for col in range(size)] for row in range(size)]
    return [[int(row == col or row + col == size - 1) for col in range(size)] for row in range(size)]


def read_matrix(name):
    print(f"\n{name} (3줄, 공백 구분. 첫 줄에 + 또는 x를 입력하면 예시 생성)")
    rows = []
    while len(rows) < 3:
        parts = input(f"{len(rows) + 1}행: ").split()
        if not rows and len(parts) == 1 and parts[0].lower() in ("+", "x"):
            label = normalize_label(parts[0])
            print(f"{label} 예시를 생성했습니다.")
            return pattern_for(3, label)
        if len(parts) != 3:
            print("입력 형식 오류: 각 줄에 3개의 숫자를 공백으로 구분해 입력하세요.")
            continue
        try:
            row = [finite_number(value) for value in parts]
        except ValueError as error:
            print(f"입력 형식 오류: {error} 같은 행을 다시 입력하세요.")
        else:
            rows.append(row)
    return rows


def mean_ms(operation, pattern, weights):
    start = perf_counter_ns()
    for _ in range(REPEATS):
        operation(pattern, weights)
    return (perf_counter_ns() - start) / (REPEATS * 1_000_000)


def show_performance(pairs):
    print(f"\n[성능 분석] MAC 한 번의 평균, {REPEATS:,}회 반복")
    print("크기     평균 시간(ms, 2D)   평균 시간(ms, 1D)   연산 횟수(N²)")
    for size, pattern, weights in pairs:
        flat_pattern = flatten(pattern)
        flat_weights = flatten(weights)
        row_time = mean_ms(mac, pattern, weights)
        flat_time = mean_ms(mac_flat, flat_pattern, flat_weights)
        print(f"{size}×{size:<4} {row_time:>16.6f} {flat_time:>18.6f} {size * size:>14}")


def run_manual():
    print("\n[1] 필터 입력")
    filter_a = read_matrix("필터 A")
    filter_b = read_matrix("필터 B")
    print("필터 A와 B를 저장했습니다.")
    print("\n[2] 패턴 입력")
    pattern = read_matrix("패턴")

    print("\n[3] MAC 결과")
    score_a, score_b = mac(pattern, filter_a), mac(pattern, filter_b)
    decision = winner(score_a, score_b, ("A", "B"))
    print(f"A 점수: {score_a:.16g} | 평균 {mean_ms(mac, pattern, filter_a):.6f} ms")
    print(f"B 점수: {score_b:.16g} | 평균 {mean_ms(mac, pattern, filter_b):.6f} ms")
    if decision == "UNDECIDED":
        print(f"판정: 판정 불가 (UNDECIDED, |A-B| < {EPSILON:g})")
    else:
        print(f"판정: {decision}")
    show_performance([(3, pattern, filter_a)])


def load_filter_pair(filters, size):
    key = f"size_{size}"
    group = filters.get(key) if isinstance(filters, dict) else None
    if not isinstance(group, dict):
        raise ValueError(f"{key} 필터 묶음이 없거나 객체가 아닙니다.")
    normalized = {}
    for raw_label, matrix in group.items():
        label = normalize_label(raw_label)
        if label in normalized:
            raise ValueError(f"{label} 필터가 중복됩니다.")
        normalized[label] = validate_matrix(matrix, size)
    if set(normalized) != {"Cross", "X"}:
        raise ValueError("Cross와 X 필터가 모두 필요합니다.")
    return normalized["Cross"], normalized["X"]


def analyze_case(case_id, case, filters, filter_errors):
    match = re.fullmatch(r"size_([1-9][0-9]*)_([1-9][0-9]*)", case_id)
    if not match:
        raise ValueError("케이스 키는 size_N_idx 형식이어야 합니다.")
    size = int(match.group(1))
    if size not in BATCH_SIZES:
        raise ValueError(f"data.json 모드에서 지원하지 않는 크기 {size}입니다.")
    if size in filter_errors:
        raise ValueError(f"필터 오류: {filter_errors[size]}")
    if not isinstance(case, dict):
        raise ValueError("케이스는 input과 expected를 가진 객체여야 합니다.")
    expected = normalize_label(case.get("expected"))
    pattern = validate_matrix(case.get("input"), size)
    cross, x = filters[size]
    score_cross, score_x = mac(pattern, cross), mac(pattern, x)
    decision = winner(score_cross, score_x, ("Cross", "X"))
    print(f"Cross 점수: {score_cross:.16g} | X 점수: {score_x:.16g}")
    if decision == expected:
        print(f"판정: {decision} | expected: {expected} | PASS")
        return None

    reason = (
        f"점수 차이가 {EPSILON:g} 미만인 동점"
        if decision == "UNDECIDED"
        else f"판정 {decision}과 기대 라벨 {expected}가 다름"
    )
    print(f"판정: {decision} | expected: {expected} | FAIL ({reason})")
    return reason


def run_batch(path):
    with path.open(encoding="utf-8") as source:
        data = json.load(source)
    if not isinstance(data, dict) or not isinstance(data.get("patterns"), dict):
        raise ValueError("data.json 최상위에는 patterns 객체가 있어야 합니다.")

    filters, filter_errors = {}, {}
    print("\n[1] 필터 로드")
    for size in BATCH_SIZES:
        try:
            filters[size] = load_filter_pair(data.get("filters"), size)
        except ValueError as error:
            filter_errors[size] = str(error)
            print(f"size_{size}: {error}")
        else:
            print(f"size_{size} 필터 로드 완료 (Cross, X)")

    print("\n[2] 패턴 분석")
    cases = data["patterns"]
    failures = []
    for case_id, case in cases.items():
        print(f"\n--- {case_id} ---")
        try:
            reason = analyze_case(case_id, case, filters, filter_errors)
        except ValueError as error:
            reason = str(error)
            print(f"FAIL ({reason})")
        if reason:
            failures.append((case_id, reason))

    show_performance([(n, pattern_for(n, "Cross"), pattern_for(n, "X")) for n in SIZES])
    print(f"\n[결과 요약]\n총 테스트: {len(cases)}개")
    print(f"통과: {len(cases) - len(failures)}개\n실패: {len(failures)}개")
    if failures:
        print("실패 케이스")
        for case_id, reason in failures:
            print(f"- {case_id}: {reason}")


def run_generator():
    while True:
        raw = input("패턴 크기 N (3~25): ").strip()
        try:
            size = int(raw)
        except ValueError:
            size = 0
        if 3 <= size <= 25:
            break
        print("3~25 사이의 정수를 입력하세요.")
    for label in ("Cross", "X"):
        print(f"\n{label} ({size}×{size})")
        for row in pattern_for(size, label):
            print(" ".join(map(str, row)))


def main():
    print("=== Mini NPU Simulator ===")
    print("1. 사용자 입력 (3×3)\n2. data.json 분석\n3. 패턴 생성기")
    try:
        while True:
            choice = input("선택: ").strip()
            if choice in ("1", "2", "3"):
                break
            print("1~3 사이의 번호를 입력하세요.")
        if choice == "1":
            run_manual()
        elif choice == "2":
            run_batch(DATA_PATH)
        else:
            run_generator()
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 중단되어 분석을 종료합니다.")
    except (OSError, ValueError) as error:
        print(f"\n분석 오류: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
