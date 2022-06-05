package 排序;

/**
 * @Description
 * @date 2018/8/19 13:49
 */
public class InsertSort {
    public void insertSort(int[] a) {
        int i, j, temp;

        for (i = 1; i < a.length; i++) {
            temp = a[i]; // 把当前待比较项付给中间量
            for (j = i; j > 0 && temp < a[j - 1]; j--) {
                // 如果待比较项小
                a[j] = a[j - 1]; // 向后移
                // 直到找到没有比比较项大的就退出当前循环
            }
            a[j] = temp;// 49
        }
        for (i = 0; i < a.length; i++) {
            System.out.print(a[i] + "\t");
        }
    }
}
