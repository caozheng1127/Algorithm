package src.src;// 题目描述
// n 只奶牛坐在一排，每个奶牛拥有 ai 个苹果，现在你要在它们之间转移苹果，使得最后所有奶牛拥有的苹果数都相同，
// 每一次，你只能从一只奶牛身上拿走恰好两个苹果到另一个奶牛上，问最少需要移动多少次可以平分苹果，如果方案不存在输出 -1。

// 输入描述:
// 每个输入包含一个测试用例。每个测试用例的第一行包含一个整数 n（1 <= n <= 100），接下来的一行包含 n 个整数 ai（1 <= ai <= 100）。

// 输出描述:
// 输出一行表示最少需要移动多少次可以平分苹果，如果方案不存在则输出 -1。

// 示例1
// 输入
// 4
// 7 15 9 5

// 输出
// 3

import java.util.*;

public class demo05_分苹果{
    public static int avg(int[] num){
        int average=0;
        int sum=0;
        int len=num.length;
        for(int i=0;i<len;i++){
            sum+=num[i];
        }
        average=sum/len;
        if(sum%len!=0){
            return -1;
        }
        for(int i=0;i<len;i++){
            int t=Math.abs((num[i]-average));
            if(t%2!=0)
                return -1;
        }
        int index=0;
        for(int i=0;i<len;i++){
            if(num[i]>average)
                index+=(Math.abs(num[i]-average)/2);
        }
        return index;
    }
    public static void main(String[] args) {
        Scanner scanner=new Scanner(System.in);
        while(scanner.hasNextInt()){
            int n=scanner.nextInt();
            int[] num=new int[n];
            for(int i=0;i<n;i++){
                num[i]=scanner.nextInt();
            }
            System.out.println(avg(num));
        }
    }
}