#!/usr/bin/env python3
# coding: utf-8
# File: xpath_demo.py
# Author: lxw
# Date: 11/24/17 5:11 PM

import re

from lxml import etree

def main():
    string = """
             <p>
             <p>
              <p>
               <span style='font-family: 微软雅黑, "Microsoft YaHei"; font-size: 12px;'>
                <span style='font-family: 微软雅黑, "Microsoft YaHei";'>
                 据市场消息，杭来湾煤矿
                </span>
                11
                <span style='font-family: 微软雅黑, "Microsoft YaHei";'>
                 月22日早上9点起，沫煤下调25元/吨，优混下调30元/吨，沫煤由464元/吨下调为439元/吨，优混由461元/吨下调为431元/吨。
                </span>
               </span>
              </p>
              <p>
              hello
              <p>
              lxw
              </p>
              </p>
            </p>
              <p>
               <span style='font-family: 微软雅黑, "Microsoft YaHei"; font-size: 12px;'>
               </span>
               <span style='font-size: 12px; font-family: 微软雅黑, "Microsoft YaHei";'>
                这次杭来湾下调煤价，是因其前期煤价相对处于高位。金鸡滩煤矿11月21日9时起，再降30元/吨，执行411元/吨。当月累计下调55元/吨；方家畔混煤下调15元，为455元；双山煤矿面煤混煤下调20元，面煤420元。以上为两票价。
               </span>
              </p>
              <p>
               <span style='font-family: 微软雅黑, "Microsoft YaHei"; font-size: 12px;'>
                目前，榆林大矿只剩下榆树湾矿没降了，估计降价就在这两天。
               </span>
              </p>
              <p>
               <span style='font-size: 12px; font-family: 微软雅黑, "Microsoft YaHei";'>
               </span>
               <span style='font-family: 微软雅黑, "Microsoft YaHei"; font-size: 12px;'>
                榆林煤价的下调表明，
               </span>
               <span style='font-family: 微软雅黑, "Microsoft YaHei"; font-size: 12px;'>
                11
               </span>
               <span style='font-family: 微软雅黑, "Microsoft YaHei"; font-size: 12px;'>
                月以来，晋陕蒙主产地煤矿复产增多，产能释放平稳，上游库存开始累积，坑口煤价出现松动。或引发区内其他煤矿跟进，产地煤价下行态势或蔓延。
               </span>
              </p>
              <p>
               
              </p>
              <p style='margin-top: 0px; margin-bottom: 0px; padding: 0px 0px 5px; line-height: 2em; color: rgb(51, 51, 51); font-family: "Microsoft Yahei", 微软雅黑, Tahoma, Arial, Helvetica, STHeiti; font-size: 12px; white-space: normal; background-color: rgb(255, 255, 255);'>
               <span style='font-size: 14px; font-family: 微软雅黑, "Microsoft YaHei";'>
                <span style="line-height: 28px;">
                 关注微信
                </span>
                <strong style="color: rgb(0, 0, 0); line-height: 28px;">
                 <span style="color: rgb(247, 150, 70);">
                  【找煤网视点】
                 </span>
                </strong>
                <span style="line-height: 28px;">
                 ，查看今日热点资讯推送（2017.11.20）：
                </span>
               </span>
              </p>
              <p style='margin-top: 0px; margin-bottom: 0px; padding: 0px 0px 5px; line-height: 2em; color: rgb(51, 51, 51); font-family: "Microsoft Yahei", 微软雅黑, Tahoma, Arial, Helvetica, STHeiti; font-size: 12px; white-space: normal; background-color: rgb(255, 255, 255);'>
               <span style="color: rgb(0, 176, 240);">
                1.
               </span>11月21-23号，找煤网与你相约@2018全国煤炭交易会——秦皇岛</p>
              <p style='margin-top: 0px; margin-bottom: 0px; padding: 0px 0px 5px; line-height: 2em; color: rgb(51, 51, 51); font-family: "Microsoft Yahei", 微软雅黑, Tahoma, Arial, Helvetica, STHeiti; font-size: 12px; white-space: normal; background-color: rgb(255, 255, 255);'>
               <span style="color: rgb(0, 176, 240);">
                2.
               </span>沿海煤市有什么变化?</p>
              <p style='margin-top: 0px; margin-bottom: 0px; padding: 0px 0px 5px; line-height: 2em; color: rgb(51, 51, 51); font-family: "Microsoft Yahei", 微软雅黑, Tahoma, Arial, Helvetica, STHeiti; font-size: 12px; white-space: normal; background-color: rgb(255, 255, 255);'>
               <span style="color: rgb(0, 176, 240);">
                3.
               </span>11月19日北方港口实际港存情况</p>
              <p style="text-align: center;">
               
              </p>
             </p>
    """
    selector = etree.HTML(string)
    result_list = selector.xpath('.//p[not(.//p)]')
    for item in result_list:
        p_string = item.xpath('string(.)').strip()
        p_string = re.sub("\s{2,}", " ", p_string)
        print(p_string)
        print("---" * 10)


def xpath_p():
    # string = "<div><p>hello world!</p></div>"
    string = """
        <p id="1">
            <p>
            hello
            </p>
            <p id="2">
                world
            </p>
        </p>
        """
    '''
    string = """
<p>
 <span>
  <p>
   <strong>
    今日观点
   </strong>
  </p>
  <p>
  </p>
  <p>
   <strong>
    期货方面，今日黑色系商品期货品种间涨跌互现，铁矿大幅增仓上涨，螺纹钢涨幅较小，其余品种收跌，其中期螺主力合约重回3800之上，后期仍有上涨空间，到位需谨慎。现货方面，今日除热轧现货稍有回落外，其余品种现货均出现上涨，螺纹钢由于资源紧俏，涨幅较大。目前来看，环保组入驻后，产量开启新一轮的下降通道，整体库存续降但降幅收窄，在途在港资源依然是后期隐患。但就目前来看，市场普遍缺规格，商家惜售情绪浓厚，预计短期国内钢价仍呈涨势。
   </strong>
  </p>
  <p>
  </p>
  <p>
   <strong>
    今日聚焦
   </strong>
  </p>
  <p>
  </p>
  <p>
   <strong>
    宏观热点
   </strong>
  </p>
  <p>
  </p>
  <p>
   <strong>
    1、【铁矿石期货主力收盘涨超3%，沪铝跌近3%】焦煤涨近3%，橡胶涨超2%，锰硅涨超1%，螺纹钢、郑煤、鸡蛋、焦炭、沪铝、白糖、硅铁、沪金、沪锌收涨。沪镍、热卷、玻璃、沪锡、郑棉、沪铜收跌。
   </strong>
  </p>
  <p>
  </p>
  <p>
   <strong>
    2、【沪指大跌2%，市场一片萧瑟，权重题材全盘尽墨】
   </strong>
  </p>
  <p>
  </p>
  <p>
   <strong>
    3、【环保部：正研究下一步排污权有偿使用相关政策】
   </strong>
  </p>
  <p>
  </p>
  <p>
   <strong>
    4、【央行发文严查“为无证机构提供支付服务行为”，明确时间表】
   </strong>
  </p>
  <p>
  </p>
  <p>
   <strong>
    原材料
   </strong>
  </p>
  <p>
  </p>
  <p>
   全国钢坯市场价格趋强调整。江苏钢坯涨40元/吨，其他地区暂稳。今日唐山钢坯直发成交一般，仓储现货3890-3900元/吨含税出库低价有成交，期钢高位震荡，但现货市场操作逐渐趋于谨慎，成品材整体成交偏弱，午后钢坯稳，现普碳150坯报3850元/吨，165矩形坯3870元/吨，20MnSi坯3950元/吨，现金含税出厂。
  </p>
  <p>
  </p>
  <p>
   国产矿主产区市场依旧风平浪静。华北区域、东北区域、华东区域价格持平及华南区域价格全部暂稳。具体来看，华北地区—唐山66%干基含税现金出厂640-650元/吨，迁西66%干基含税现金出635-645元/吨，迁安66%干基含税现金出厂655-665元/吨；遵化66%干基含税现金出厂650-660元/吨。
  </p>
  <p>
  </p>
  <p>
   进口矿现货市场表现强势，贸易商早盘报价上调10元/吨左右，挺价意愿明显，其中高品粉矿：主流PB粉报价500元/吨，中低品粉矿：主流超特粉报价295元/吨。在期货盘面维持较大涨幅的提振下，钢企采购热情回暖，询盘依旧以MNP等主流高品粉矿为主，现货市场成交较昨日好转，价格坚挺上涨10元/吨左右。
  </p>
  <p>
  </p>
  <p>
   <img/>
  </p>
  <p>
  </p>
  <p>
   <strong>
    钢材现货
   </strong>
  </p>
  <p>
  </p>
  <p>
   <strong>
    建筑钢材：
   </strong>
   今日国内建筑钢材价格继续上涨。具体价格来看，全国25个主要城市均价4351元/吨，较昨日价格上涨31元/吨，其中华中、华北。西南地区涨幅最为明显，涨幅在50-70元/吨，华东、华北、东北等地涨幅在10-50元/吨不等。具体市场来看，今日期螺主力继续强势运行，下游需求也表现良好，故在库存偏低的情况下商家继续惜售拉涨，整体氛围积极。目前来看，市场库存继续大幅下降，市场心态良好，预计明日国内建筑钢材价格或继续上涨。
  </p>
  <p>
  </p>
  <p>
   <strong>
    热轧板卷：
   </strong>
   今日全国24个主要城市热轧价格震荡运行，3.0热轧板卷全国均价4273元/吨，较上一交易日下跌3元/吨，4.75热轧板卷全国均价4216元/吨，较上一交易日下跌2元/吨。今日期卷震荡下行，市场观望情绪较浓，商家报价盘整运行。目前市场需求仍没有明显好转，高价资源成交困难，不过全国热轧市场库存仍在下降，商家销售压力不大，因此也不愿过低销售。今日钢坯市场价格维持平稳，现普碳坯价格在3850元/吨。综合来看，预计明日热轧市场价格震荡运行。
  </p>
  <p>
  </p>
  <p>
   <strong>
    中厚板：
   </strong>
   今日国内中厚板市场价格横盘整理，全国23个主要城市20mm中厚板均价4195元/吨，较上一交易日上涨4元/吨。北方坯料价格盘整运行，伴随着期货市场的震荡走势，商家继续看涨情绪有所减弱。此外随着成交方面的回落，加重了下游及商家谨慎操作的心态，并且下游采购单位对于价格的上涨并不买单，近阶段大多采取按需采购的策略。目前上游生产企业继续维持对市场的挺价态度，且现货资源整体库存压力不大，因此综合预计明日国内中厚板市场价格或继续维持横盘整理局势。
  </p>
  <p>
  </p>
  <p>
   <strong>
    冷轧板卷：
   </strong>
   今日全国冷轧价格弱势上涨。价格方面：1.0全国冷轧均价4832元/吨，较上个工作日上涨6元/吨。主要市场价格方面：上海市场1.0mm武钢卷板报价4780元/吨，广州市场1.0mm鞍钢卷报价4910，天津市场1.0mm鞍钢天铁卷报价4690吨。市场方面：期货市场偏强震荡，现货市场库存较上周减少，压力减轻，市场主流大户心态总体持稳。当前价位较高，市场成交明显减弱。预计明日冷轧价格震荡偏弱运行。
  </p>
  <p>
  </p>
  <p>
   <strong>
    型材：
   </strong>
   今日国内型钢市场价格盘整趋强。首先迫于钢厂调价带来的成本压力，底部价格延续涨势，小幅上涨20-30元/吨，而由于近两日随着价格不断上涨成交情况反而一直下滑，因此主流市场价格则持稳观望。虽然周初至今型材整体走势趋强运行，但现货依旧徘徊在倒挂边缘，而随着冬季限产以及厂价过高等因素，后续的到货资源也相应减少，库存支撑逐渐体现出来。不过市场在面临资金压力过大且成交乏力的情况下，也将选择理性控制整体涨幅。因此预计明日国内型钢市场价格或以高位盘整为主。
  </p>
  <p>
  </p>
  <p>
   <strong>
    钢管：
   </strong>
   今日国内钢管价格继续小幅上涨。分品种来看，焊管4寸3.75mm全国均价4617元/吨，较上一交易日上涨8元/吨；镀锌管4寸3.75mm全国均价5335元/吨，较上一交易日上涨9元/吨；无缝管108*4.5mm全国均价5566元/吨，较上一交易日上涨9元/吨。管厂方面，天津友发、利达、君诚、邯郸正大暂稳，临沂主流无缝管厂热轧报价5250元/吨。焊管、镀锌管方面，市场成交一般，北方主流焊管厂多错峰生产，资源供应相对紧张。无缝管方面，市场成交依旧疲软，山东管厂在成本支撑下提价出厂，社会库存有所下降，高价位管材开始流入市场。预计明日钢管市场价格仍将震荡趋强运行。
  </p>
  <p>
  </p>
  <p>
   <img/>
  </p>
  <p>
  </p>
  <p>
   <strong>
    期货：
   </strong>
   国内黑色系商品期货冲高回落，主力资金大幅流入，成交小幅放量，市场火热。整体方面，今日黑色系资金轮动，沉迷许久的铁矿石高开高走大幅拉涨，主力顺势移仓05合约，但近期双焦涨幅过大，早盘随着获利盘的陆续止盈，导致市场做多情绪熄灭，尾盘小幅下跌，短期回调将至。从技术层面来看，以期螺为例：夜盘期螺横盘整体，高位震荡，主力大幅增仓；早盘期螺震荡走高，冲击前期高点，3850附近承压过重，二次上冲均迅速回落；午后期螺保持横盘整理，观望情绪浓厚。从日线上面来看，期螺今日收长上影线，今日主力增仓滞涨，夜盘恐有所回调，短期行情看震荡为主，操作方面建议回落做多，前高附近可小仓位做空，注意仓位。（以上操作仅供参考！）
  </p>
  <p>
  </p>
  <p>
   <img/>
  </p>
 </span>
 <br/>
</p>

    """
    string = """
    <p>
    lxw
      <p>
       <strong>
        今日观点
       </strong>
      </p>
     <br/>
    </p>

        """
    '''
    selector = etree.HTML(string)
    # result_list = selector.xpath('./p')   # NO
    result_list = selector.xpath('//p')    # OK
    # result_list = selector.xpath('p')    # NO
    print(result_list)
    for item in result_list:
        p_string = item.xpath('string(.)').strip()
        print(p_string)
        print("---" * 10)


if __name__ == '__main__':
    # main()
    xpath_p()