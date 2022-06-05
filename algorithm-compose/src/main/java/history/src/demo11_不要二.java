package history.src;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

/**
 * 题目描述
 *  二货小易有一个W*H的网格盒子，网格的行编号为0~H-1，网格的列编号为0~W-1。每个格子至多可以放一块蛋糕，
 *  任意两块蛋糕的欧几里得距离不能等于2。
 *  对于两个格子坐标(x1,y1),(x2,y2)的欧几里得距离为:
 *  ( (x1-x2) * (x1-x2) + (y1-y2) * (y1-y2) ) 的算术平方根
 *  小易想知道最多可以放多少块蛋糕在网格盒子里。
 *
 *  输入描述:
 *  每组数组包含网格长宽W,H，用空格分割.(1 ≤ W、H ≤ 1000)
 *
 *  输出描述:
 *  输出一个最多可以放的蛋糕数
 *
 *  示例1
 *  输入
 *  3 2
 *
 *  输出
 *  4
  */


public class demo11_不要二 {

    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        String str = null;
        String[] string;
        int w, h, count = 0;
        while((str = reader.readLine()) != null) {
            string = str.split(" ");
            w = Integer.parseInt(string[0]);
            h = Integer.parseInt(string[1]);
            if(w % 4 == 0 || h % 4 ==0) {
                count = w * h / 2;
            }
            else if(w % 2 == 0 && h % 2 == 0) {
                count = (w * h / 4 + 1) / 2;
            }
            else {
                count = (w * h) / 2 + 1;
            }
            System.out.println(count);
        }
    }
}