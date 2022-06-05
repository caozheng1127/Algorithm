package history.leetcode;

public class 跳台阶 {
    public int JumpFloor(int target) {
        int step = 0;
        if(target==0){
            step = 0;
        }
        if(target==1){
            step = 1;
        }
        if (target==2){
            step=2;
        }
        if (target>2){
            step= JumpFloor(target-1) + JumpFloor(target-2);
        }
        return step;
    }

    public class Solution {
        public int JumpFloor(int target) {
            if(target <= 0) return 0;
            if(target == 1) return 1;
            if(target == 2) return 2;
            int one = 1;
            int two = 2;
            int result = 0;
            for(int i = 2; i < target; i++){
                result = one+ two;
                one = two;
                two = result;
            }
            return result;
        }
    }
}
