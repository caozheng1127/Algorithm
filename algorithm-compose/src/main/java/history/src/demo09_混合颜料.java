package history.src;// 题目描述
// 你就是一个画家！你现在想绘制一幅画，但是你现在没有足够颜色的颜料。为了让问题简单，我们用正整数表示不同颜色的颜料。
// 你知道这幅画需要的n种颜色的颜料，你现在可以去商店购买一些颜料，但是商店不能保证能供应所有颜色的颜料，
// 所以你需要自己混合一些颜料。混合两种不一样的颜色A和颜色B颜料可以产生(A XOR B)这种颜色的颜料
// (新产生的颜料也可以用作继续混合产生新的颜色,XOR表示异或操作)。本着勤俭节约的精神，你想购买更少的颜料就满足要求，
// 所以兼职程序员的你需要编程来计算出最少需要购买几种颜色的颜料？

// 输入描述:
// 第一行为绘制这幅画需要的颜色种数n (1 ≤ n ≤ 50)
// 第二行为n个数xi(1 ≤ xi ≤ 1,000,000,000)，表示需要的各种颜料.

// 输出描述:
// 输出最少需要在商店购买的颜料颜色种数，注意可能购买的颜色不一定会使用在画中，只是为了产生新的颜色。

// 示例1
// 输入
// 3
// 1 7 3

// 输出
// 3


// 这题我们首先采取的是暴力求解办法，是采取的是对于选取的颜色限购买最小的那个颜色，
// 然后在进行选取需要颜色中不存在与我们已经购买的颜色和购买颜色产生的颜色的颜色进行购买，知道产生所有颜色为主，
// 这种暴力求解方法的算法复杂度过大，不能够通过所有的测试用例。所以我们就要寻找规律。
// 对于我们在面试的时候如果不能够快速找寻出问题的规律的话，我们就选取暴力求解办法，虽然不能通过所有的测试用例，
// 但是却能够通过一部分用例，达到部分求解。

// 本题的规律就是对于哪些大的数，如果没有存在与他高位相同的数的时候，那么这个颜色就必须购买，
// 因为要想生成这种颜色就必须有一个与这个颜色高位相同的颜色存在，所以我们直接购买这个颜色本身是最直接的。
// 对于存在高位颜色相同的数的时候，这两个高位相同的颜色异或获得的颜色肯定小于这个颜色，
// 那么我们就将这个小的颜色添加到这个序列中再重新进行判断计算。这是我们的数组会更加有利。这就是这个题的规律。
// 其实这里在以后如果遇到异或相关方面的问题的时候我们都可以从数位角度进行考虑。


//import java.io.*;
//import java.util.*;
//
//public class demo09_混合颜料 {
//    public static void main(String[] args) throws IOException {
//        BufferedReader bf = new BufferedReader(new InputStreamReader(System.in));
//        String str = bf.readLine();
//        int n = Integer.parseInt(str);
//        ArrayList<Integer> list = new ArrayList<>();
//        str = bf.readLine();
//        String[] string = str.split(" ");
//        for (int i = 0; i < n; i++) {
//            list.add(Integer.parseInt(string[i]));
//        }
//        int sum = 1;
//        ArrayList<Integer> mylist = new ArrayList<>();
//        mylist.add(list.get(0));
//        list.remove(0);
//        while (list.size() != 0) {
//            int temp = list.get(0);
//            list.remove(0);
//            sum++;
//            int t = mylist.size();
//            mylist.add(temp);
//            for (int i = 0; i < t; i++) {
//                int q = mylist.get(i) ^ temp;
//                if (!mylist.contains(q)) {
//                    mylist.add(q);
//                }
//                if (list.size() == 0) {
//                    break;
//                }
//                if (list.contains(q)) {
//                    int index = list.indexOf(q);
//                    list.remove(index);
//                }
//            }
//        }
//        System.out.println(sum);
//    }
//}


import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;

public class demo09_混合颜料 {
    public static void main(String[] args) throws IOException {
        BufferedReader bf = new BufferedReader(new InputStreamReader(System.in));
        String str = null;
        while ((str = bf.readLine()) != null) {
            int n = Integer.parseInt(str);
            str = bf.readLine();
            String[] string = str.split(" ");
            long[] array = new long[n];
            for (int i = 0; i < n; i++) {
                array[i] = Long.parseLong(string[i]);
            }
            Arrays.sort(array);
            int number = 1;
            int last = array.length - 1;
            while (last > 0 && array[last - 1] != 0) {
                if (highBit(array[last - 1]) == highBit(array[last])) {
                    array[last - 1] = array[last - 1] ^ array[last];
                    Arrays.sort(array);
                } else {
                    number++;
                    last--;
                }
            }
            System.out.println(number);
        }
    }

    public static int highBit(long n) {
        int cnt = 0;
        while (n > 0) {
            n = n / 2;
            cnt++;
        }
        return cnt;
    }
}