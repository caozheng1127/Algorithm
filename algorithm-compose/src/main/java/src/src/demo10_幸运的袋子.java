package src.src;// 题目描述
// 一个袋子里面有n个球，每个球上面都有一个号码(拥有相同号码的球是无区别的)。
// 如果一个袋子是幸运的当且仅当所有球的号码的和大于所有球的号码的积。

// 例如：如果袋子里面的球的号码是{1, 1, 2, 3}，这个袋子就是幸运的，因为1 + 1 + 2 + 3 > 1 * 1 * 2 * 3
// 你可以适当从袋子里移除一些球(可以移除0个,但是别移除完)，要使移除后的袋子是幸运的。
// 现在让你编程计算一下你可以获得的多少种不同的幸运的袋子。

// 输入描述:
// 第一行输入一个正整数n(n ≤ 1000)
// 第二行为n个数正整数xi(xi ≤ 1000)

// 输出描述:
// 输出可以产生的幸运的袋子数

// 示例1
// 输入
// 3
// 1 1 1

// 输出
// 2

import java.util.*;

//public class demo10_幸运的袋子 {
//    public static void main(String args[]) {
//        Scanner sc = new Scanner(System.in);
//        while (sc.hasNext()) {
//            int n = sc.nextInt();
//            int[] a = new int[n];
//            for (int i = 0; i < n; i++) {
//                a[i] = sc.nextInt();
//            }
//            System.out.println(output(a));
//        }
//
//    }
//
//    private static int output(int[] num) {
//        int cnt = 0;
//        ArrayList<Integer> list = new ArrayList<>();
//        for (int i = 0; i < num.length; i++) {
//            list.add(num[i]);
//        }
//        while (list.size() > 1) {
//            for (int i = 0; i < list.size(); i++) {
//                for (int j = 0; j < list.size(); j++) {
//                    ArrayList<Integer> listCopy = (ArrayList<Integer>) list.clone();
//                    listCopy.remove(listCopy.get(j));
//                    if (judge(list)) {
//                        cnt++;
//                    }
//                }
//                list.remove((list.get(i)));
//            }
//        }
//        return cnt;
//    }
//
//    public static boolean judge(ArrayList arrayList) {
//        int sum = 0;
//        int mul = 1;
//        boolean k = false;
//        for (int i = 0; i < arrayList.size(); i++) {
//            sum = sum + (int) arrayList.get(i);
//            mul = mul * (int) arrayList.get(i);
//        }
//        if (sum > mul) {
//            k = true;
//        }
//        return k;
//    }
//}

public class demo10_幸运的袋子 {

    public static void main(String[] args) {
        Scanner in=new Scanner(System.in);
        while(in.hasNext()){
            int n=in.nextInt();
            int[] arr=new int[n];
            for(int i=0;i<n;i++){
                arr[i]=in.nextInt();
            }
            Arrays.sort(arr);
            System.out.println(find(arr,0,0,1));
        }
    }
    private static int find(int[] arr, int index, long sum, long mul) {
        int count=0;
        for(int i=index;i<arr.length;i++){
            sum+=arr[i];
            mul*=arr[i];
            if(sum>mul)
                count+=1+find(arr,i+1,sum,mul);
            else if(arr[i]==1)
                count+=find(arr,i+1,sum,mul);
            else
                break;
            sum-=arr[i];
            mul/=arr[i];
            while(i<arr.length-1&&arr[i]==arr[i+1])
                i++;
        }
        return count;
    }

}