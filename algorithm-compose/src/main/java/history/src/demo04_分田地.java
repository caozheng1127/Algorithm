package history.src;// 题目描述
// 牛牛和 15 个朋友来玩打土豪分田地的游戏，牛牛决定让你来分田地，地主的田地可以看成是一个矩形，每个位置有一个价值。
// 分割田地的方法是横竖各切三刀，分成 16 份，作为领导干部，牛牛总是会选择其中总价值最小的一份田地，
// 作为牛牛最好的朋友，你希望牛牛取得的田地的价值和尽可能大，你知道这个值最大可以是多少吗？

// 输入描述:
// 每个输入包含 1 个测试用例。每个测试用例的第一行包含两个整数 n 和 m（1 <= n, m <= 75），表示田地的大小，
// 接下来的 n 行，每行包含 m 个 0-9 之间的数字，表示每块位置的价值。

// 输出描述:
// 输出一行表示牛牛所能取得的最大的价值。

// 示例1
// 输入
// 4 4
// 3332
// 3233
// 3332
// 2323

// 输出

// 2

import java.util.Scanner;

public class demo04_分田地{
    static int n;
    static int m;
    static int[][] data;
    public static void main(String[] args){
        Scanner scan = new Scanner(System.in);
        while(scan.hasNext()){
            n = scan.nextInt();
            m = scan.nextInt();
            data = new int[n+1][m+1];
            for(int i=1;i<=n;i++){
                String str = scan.next();
                char[] strchar = str.toCharArray();
                for(int j=1;j<=m;j++){
                    data[i][j] = strchar[j-1]-'0';
                }
            }
            for(int i=1;i<=n;i++){
                for(int j=1;j<=m;j++){
                    data[i][j]=data[i-1][j]+data[i][j-1]-data[i-1][j-1]+data[i][j];
                }
            }
            int l = 0;
            int r = data[n][m];
            int mid = r = (l+r)>>3;
            int res = 0;
            while(l<=r){
                mid = (l+r)>>1;
                if(check(mid)){
                    l = mid+1;
                    res = mid;
                }else{
                    r = mid-1;
                }
            }
            System.out.println(res);
        }
    }
    private static boolean check(int mid) {
        for(int i=1;i<=n-3;i++){
            for(int j=i+1;j<=n-2;j++){
                for(int k = j+1;k<=n-1;k++){

                    int last = 0;
                    int count = 0;
                    for(int h = 1;h<=m;h++){
                        int sum1 = getSum(0,i,last,h);
                        int sum2 = getSum(i,j,last,h);
                        int sum3 = getSum(j,k,last,h);
                        int sum4 = getSum(k,n,last,h);
                        if(mid<=sum1&&mid<=sum2&&mid<=sum3&&mid<=sum4){
                            last = h;
                            count++;
                        }
                    }
                    if(count>=4){
                        return true;
                    }
                }
            }
        }
        return false;
    }
    private static int getSum(int startx, int x, int starty, int y) {
        return data[x][y]-data[startx][y]-data[x][starty]+data[startx][starty];
    }
}
