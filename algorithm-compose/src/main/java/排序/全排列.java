package 排序;

import java.util.Arrays;
import org.apache.commons.lang3.ArrayUtils;

/**
 * @Description
 * @date 2018/8/19 16:43
 */
public class 全排列 {
    private static void core(int[] array) {
        int length = array.length;
        fullArray(array, 0, length - 1);
    }

    private static boolean swapAccepted(int[] array, int start, int end) {
        for (int i = start; i < end; i++) {
            if (array[i] == array[end]) {
                return false;
            }
        }
        return true;
    }

    private static void fullArray(int[] array, int cursor, int end) {
        if (cursor == end) {
            System.out.println(Arrays.toString(array));
        } else {
            for (int i = cursor; i <= end; i++) {
                if (!swapAccepted(array, cursor, i)) {
                    continue;
                }
                ArrayUtils.swap(array, cursor, i);
                fullArray(array, cursor + 1, end);
                ArrayUtils.swap(array, cursor, i); // 用于对之前交换过的数据进行还原
            }
        }
    }
}
