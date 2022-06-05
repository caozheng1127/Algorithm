package src.src;// 题目描述
// 给定一个 n 行 m 列的地牢，其中 '.' 表示可以通行的位置，'X' 表示不可通行的障碍，牛牛从 (x0 , y0 ) 位置出发，
// 遍历这个地牢，和一般的游戏所不同的是，他每一步只能按照一些指定的步长遍历地牢，要求每一步都不可以超过地牢的边界，
// 也不能到达障碍上。地牢的出口可能在任意某个可以通行的位置上。牛牛想知道最坏情况下，他需要多少步才可以离开这个地牢。

// 输入描述:
// 每个输入包含 1 个测试用例。每个测试用例的第一行包含两个整数 n 和 m（1 <= n, m <= 50），表示地牢的长和宽。
// 接下来的 n 行，每行 m 个字符，描述地牢，地牢将至少包含两个 '.'。接下来的一行，包含两个整数 x0, y0，
// 表示牛牛的出发位置（0 <= x0 < n, 0 <= y0 < m，左上角的坐标为 （0, 0），出发位置一定是 '.'）。
// 之后的一行包含一个整数 k（0 < k <= 50）表示牛牛合法的步长数，接下来的 k 行，
// 每行两个整数 dx, dy 表示每次可选择移动的行和列步长（-50 <= dx, dy <= 50）

// 输出描述:
// 输出一行一个数字表示最坏情况下需要多少次移动可以离开地牢，如果永远无法离开，输出 -1。以下测试用例中，
// 牛牛可以上下左右移动，在所有可通行的位置.上，地牢出口如果被设置在右下角，牛牛想离开需要移动的次数最多，为3次。

// 示例1

// 输入
// 3 3
// ...
// ...
// ...
// 0 1
// 4
// 1 0
// 0 1
// -1 0
// 0 -1

// 输出
// 3

import java.util.LinkedList;
import java.util.Queue;
import java.util.Scanner;

public class demo02_地牢逃脱 {
    public static int BFS(boolean[][] maze,boolean[][] visit,int[] start,int[] end,int[][] dirs){
        Queue<int[]> queue = new LinkedList<>();
        int [] now,next;
        int [] node = new int []{start[0],start[1],0};
        queue.add(node);
        visit[start[0]][start[1]] = true;
        int maxStep = 0; //整个队列中走的步数最多的步数
        while (!queue.isEmpty()){
            now = queue.poll();
            //if (now[0] == end[0] && now[1] == end[1]) return now[2];
            if(maxStep < now[2]) maxStep = now[2];
            for(int[] dir:dirs){
                next = new int[]{dir[0]+now[0],dir[1]+now[1],now[2]+1};
                if (next[0]<0||next[0]>=maze.length || next[1] < 0|| next[1] >= maze[0].length
                        ||visit[next[0]][next[1]] == true||maze[next[0]][next[1]] == true){
                    continue;
                }
                queue.add(next);
                visit[next[0]][next[1]] = true;
            }
        }
        //有没走到的返回-1
        for (int i = 0;i < maze.length; i++){
            for (int j = 0;j < maze[0].length; j++){
                if (visit[i][j] ==false && maze[i][j] == false) return -1;
            }
        }

        return  maxStep;
    }
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        while(sc.hasNext()){
            int n = sc.nextInt();
            int m = sc.nextInt();
            boolean[][] maze = new boolean[n][m];
            for(int i = 0; i < n; i++){
                String s = sc.next();
                for(int j = 0; j < m; j++){
                    if(s.charAt(j) != '.') maze[i][j] = true;
                }
            }
            int []start = new  int[]{sc.nextInt(),sc.nextInt()};
            int ndir = sc.nextInt();
            int [][] dirs = new int[ndir][2];
            for(int i = 0; i < ndir; i++){
                dirs[i][0] = sc.nextInt();
                dirs[i][1] = sc.nextInt();
            }
            boolean[][] visit = new boolean[n][m];
            System.out.println(BFS(maze,visit,start,new int[]{n-1,m-1},dirs));
        }
        sc.close();
    }
}