---
title: ANN Programming Assignment2
---

:::info
系級：資工三乙
學號：406262084
姓名：梁博全
:::


# Statistic

* total time: 3188 Sec
* Termination condition: 
    * T(RMSE) < 0.2
    * epoch limitation < 80000


* time(sec):

| LR\Hidden  |  1     | 4        | 8     | 12     | 
| ---------  | ---    | ----     |---    |---     |
| 2          | 606.23 | 2.79     | 1.79  | 4.47   |
| 1          | 641.04 | 1.21     | 0.99  | 1.62   |
| 0.5        | 675.14 | 1249.86  | 1.058 | 1.834  |


* training accuracy:

| LR\Hidden  |  1    | 4      | 8      | 12     | 
| ---------  | ---   | ----   |---     |---     |
| 2          | 66.6% | 96.6%  | 96.6%  | 96.6%  |
| 1          | 66.6% | 96.6%  | 96.6%  | 96.6%  |
| 0.5        | 90%   | 88.3%  | 96.6%  | 96.6%  |

* testing accuracy

| LR\Hidden  |  1    | 4      | 8      | 12     | 
| ---------  | ---   | ----   |---     |---     |
| 2          | 66.6% | 96.6%  | 100%   | 100%   |
| 1          | 66.6% | 96.6%  | 96.6%  | 100%   |
| 0.5        | 93.3% | 90%    | 96.6%  | 96.6%  |



# Analysis

### different hidden number

* 在相同的`learning rate`下，hidden number越高，準確率也越大，所花費時間訓練也會有所提高


### differrnt learning rate

* 在相同`hidden`下, learning rate在`2`的情況基本上花費時間比較高，再來是`0.5`, `1`

* 由於`learning rate`為2時，更新的幅度較大，有可能會更新過頭而需要更新回來，導致花較久時間找到min
* 而`0.5`時，又因為降低了rate，幅度較小，所以也會花多異點時間更新

### special case

* `hidden=4` `learning rate=0.5`的情況:
    *  根據統計資料發現，這筆是唯一一個在相同`learning rate`，準確率卻較低的情況。而訓練時間卻也極高。
    * 原因: 此case找到local min，且由於`learning rate`設為`0.5`，無法跳脫出local min 去找到更小的誤差。


### better performance

* `hidden = 12` `learning rate = 2 or 1`有比較好的表現
* `learning rate = 0.5`準確率平均起來最高






