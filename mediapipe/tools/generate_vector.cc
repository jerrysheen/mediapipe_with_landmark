#include "generate_vector.h"

using namespace std;
vector<vector<double>> generate_vector(string str, int image_x, int image_y){
            // regular expression 取值    
        std::regex reg("[\\\n|\\s]");
        // output: hello, world!
        str = regex_replace(str, reg, "");
        //cout << str << endl;
        //cout << str << endl;
        smatch result;
        //regex pattern("\\{([^}]*)\\}");    //{中的内容}
        regex pattern("x([^z]*)");
        regex_search(str, result, pattern);    
        //遍历结果
        
        //迭代器声明
        string::const_iterator iterStart = str.begin();
        string::const_iterator iterEnd = str.end();
        regex pattern1("[\\d|.]+[^y]");
        regex pattern2("y:.+");
        smatch xcord;
        smatch ycord;
        double x;
        double y; 
        vector<vector<double>> cordinate_collection;
        // result 0 x:0.74297905y:0.690012038
        while (regex_search(iterStart, iterEnd, result, pattern))
        {
            vector<double> pair;
            str = result[0];
            // 0.742979获取
            //cout << xcord[0] << endl;
            regex_search(str, xcord, pattern1);
            regex_search(str, ycord, pattern2);
            //cout << xcord.str(0) << endl;
            //cout << ycord.str(0).substr(2,ycord.str(0).length()) << endl;
            // rows 480
            // cols 640
            x = stod(xcord.str(0)) * image_x;
            y = stod(ycord.str(0).substr(2,ycord.str(0).length())) * image_y;
            pair.push_back(x);
            pair.push_back(y);
            cordinate_collection.push_back(pair);
            iterStart = result[0].second;
        }   
        for(vector<double> cordinate : cordinate_collection){
              cout << cordinate[0] << " " <<cordinate[1] << endl;
        }

        return cordinate_collection;
}