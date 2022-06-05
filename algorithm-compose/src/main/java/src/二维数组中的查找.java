package src;//从左下角元素往上查找，右边元素是比这个元素大，上边是的元素比这个元素小。于是，target比这个元素小就往上找，
// 比这个元素大就往右找。如果出了边界，则说明二维数组中不存在target元素。

public class 二维数组中的查找 {
    public boolean Find(int target, int [][] array) {
        int rows = array.length;
        int cols = array[0].length;
        int i=rows-1,j=0;
        while(i>=0 && j<cols){
            if(target<array[i][j])
                i--;
            else if(target>array[i][j])
                j++;
            else
                return true;
        }
        return false;
    }
}
