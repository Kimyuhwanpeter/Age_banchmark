# Age_banchmark
* 우선 데이터는 다 나누었음 (Morph, UTK, AFAD, and AAF)
* 테스트를 진행할 때는 WM (White male), WF (White female), BM (Black male), BF (Black female), AM (Asian male), AF (Asian female)로 각각 측정할 것 (한번에 측정 가능)
* train-3,449; test-3,449 (이 안에서 WM, WF, BM, BF, AM, AF 을 각각 나누어야 함)
* 학습코드는 내일 오전중 수정하고 진행할 것 (DEX, MV, CORAL, CCNNAE 총 네개)
* 당연한것이지만 Morph dataset과 같이 mugshot 이미지가 성능이 좋음

## DEX (pre-trained IMDB-WIKI 는 사용x)
|     Train/test    |  WM  |  WF  |  BM  |  BF  |  AM  |  AF  |
| :---------------: | :--: | :--: | :--: | :--: | :--: | :--: |
| Morph (20 epochs) | 3.78 | 4.72 | 3.85 | 4.56 |  -   |  -   |
|   UTK (18 epochs) | 5.98 | 5.88 | 5.31 | 5.36 |  -   |  -   |
|        AFAD       |  -   |  -   |  -   |  -   | 예정 | 예정  |
|        AAF        |  -   |  -   |  -   |  -   | 예정 | 예정  |

## MV (pre-trained IMDB-WIKI 는 사용x)
|     Train/test    |  WM  |  WF  |  BM  |  BF  |  AM  |  AF  |
| :---------------: | :--: | :--: | :--: | :--: | :--: | :--: |
| Morph (20 epochs) | 3.87 | 4.28 | 3.67 | 4.15 |  -   |  -   |
|   UTK (29 epochs) | 5.90 | 5.61 | 4.98 | 5.08 |  -   |  -   |
|        AFAD       |  -   |  -   |  -   |  -   | 예정 | 예정  |
|        AAF        |  -   |  -   |  -   |  -   | 예정 | 예정  |

## CORAL (pre-trained 사용 o)
|     Train/test    |  WM  |  WF  |  BM  |  BF  |  AM  |  AF  |
| :---------------: | :--: | :--: | :--: | :--: | :--: | :--: |
| Morph (123 epochs)| 3.96 | 4.05 | 3.64 | 4.36 |  -   |  -   |
| UTK (969 epochs)  | 5.37 | 5.53 | 5.02 | 5.45 |  -   |  -   |
|        AFAD       |  -   |  -   |  -   |  -   | 예정 | 예정  |
|        AAF        |  -   |  -   |  -   |  -   | 예정 | 예정  |

## DLDL-V2 (pre-trained 사용 x) --> Morph 실험중 (지금 learning rate decay를 맞춰보는중)
|     Train/test    |  WM  |  WF  |  BM  |  BF  |  AM  |  AF  |
| :---------------: | :--: | :--: | :--: | :--: | :--: | :--: |
|        Morph      |  -   |  -   |  -   |  -   |  -   |  -   |
|        UTK        |  -   |  -   |  -   |  -   |  -   |  -   |
|        AFAD       |  -   |  -   |  -   |  -   | 예정 | 예정  |
|        AAF        |  -   |  -   |  -   |  -   | 예정 | 예정  |

