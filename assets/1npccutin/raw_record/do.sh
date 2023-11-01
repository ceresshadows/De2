#!/bin/bash

# 迭代从0到199
for i in {0..82}
do
    # 执行第一种cyber_record命令
    cyber_record echo -f record_${i}.record.00000 -t /apollo/localization/pose > ${i}_pose.txt
    
    # 执行第二种cyber_record命令
    cyber_record echo -f record_${i}.record.00000 -t /apollo/control > ${i}_control.txt
    
    # 执行第三种cyber_record命令
    cyber_record echo -f record_${i}.record.00000 -t /apollo/perception/obstacles > ${i}_obstacle.txt
done
