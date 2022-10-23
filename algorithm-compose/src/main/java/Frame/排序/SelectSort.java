package Frame.排序;

/**
 * @Description
 * @date 2018/8/19 13:50
 */
public class SelectSort {
    public static void selectSort(int[] a) {
        int position = 0;
        for (int i = 0; i < a.length; i++) {
            int j = i + 1;
            position = i;
            int temp = a[i];
            for (; j < a.length; j++) {
                if (a[j] < temp) {
                    temp = a[j];
                    position = j;
                }
            }
            a[position] = a[i];
            a[i] = temp;
        }
        for (int i = 0; i < a.length; i++)
            System.out.print(a[i] + "\t");
    }
}
