import visualization.barPlot as bar
import visualization.boxenPlot as boxen
import visualization.boxPlot as box
import visualization.catPlot as cat
import visualization.clusterMap as cluster
import visualization.countPlot as count
import visualization.dataDistributionPlot as dataDist
import visualization.distPlot as dist
import visualization.heatMap as heat
import visualization.jointPlot as joint
import visualization.kdePlot as kde
import visualization.linePlot as line
import visualization.lmPlot as lm
import visualization.pairGrid as pairGrid
import visualization.pairPlot as pair
import visualization.pointPlot as point
import visualization.regPlot as reg
import visualization.relationPlot as relation
import visualization.residPlot as resid
import visualization.rugPlot as rug
import visualization.scatterPlot as scatter
import visualization.stripPlot as strip
import visualization.swarmPlot as swarm
import visualization.violinPlot as violin
import visualization.scatterPlotWithEncircle as scatterEncircle
import visualization.divergingBars as divBars
import visualization.divergingBarTexts as divBarText
import visualization.divergingLollipop as divLolly
import visualization.divergingDotPlot as divDot
import visualization.areaChart as area
import visualization.slopePlot as slope
import visualization.dumbellPlot as dumbell
import visualization.densityCurveWithHistogramPlot as densityCurve
import visualization.distrubutedDotPlot as distDot
import visualization.densityPlot as density
import visualization.dotBoxPlot as dotBox
import visualization.tufteBoxPlot as tufte
import visualization.peakTroughTimeSeries as peakTrough
import visualization.crossCorelation as cross
import visualization.timeSeriesDecomposition as decompdataPath
import visualization.multipleTimeSeries as multiple
import visualization.seasonalPlot as seasonal
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from DimensionalityReduction import PrincipleComponentAnalysis as PCA

def chartPlotter(dataPath,data,chartsList,inputDict,splitPercentage,outputPath,pcaComp=0,dataType='train'):
    compressedPath = outputPath+os.sep+'data_%s.csv.gz' % inputDict['ticketId']
    data.to_csv(compressedPath,compression='gzip')
    #print('data split started .. ')
    X_train, data = train_test_split(data, test_size=round(splitPercentage/100,2), random_state=42)
    if(pcaComp > 0):
        data = PCA.principalComponentConverter(data,pcaComp)
    visual_data = pd.DataFrame(columns=['TicketId','Data','PlotType','x','y','hue','Category','confidence_interval','lowess','lineWidth','timeVariantCol','timeCol','yearColumn1',
                                        'yearColumn2','timeVariantCol1','timeVariantCol2','categoryCol','categoryList','categoryVariantCol','dateColumn','timeVariantColumns','compressedDataPath'])
    TicketIdList,DataList,PlotTypeList,xList,yList,hueList,CategoryList,confidence_intervalList,lowessList,lineWidthList,timeVariantColList,timeColList = [],[],[],[],[],[],[],[],[],[],[],[]
    yearColumn1List,yearColumn2List,timeVariantCol1List,timeVariantCol2List,categoryColList,categoryListList,categoryVariantColList = [],[],[],[],[],[],[]
    dateColumnList,timeVariantColumnsList,compressedDataPathList = [],[],[]

    for chartType in chartsList:

        print('chart type : ',chartType)
        print('initial values : ',list(visual_data['TicketId']))
        print('current value : ',[inputDict.get('ticketId', [''])])

        TicketIdList.append([inputDict.get('ticketId', [''])])
        print('a : ',[inputDict.get('ticketId', [''])])
        print('after processing : ',list(visual_data['TicketId']))
        DataList.extend([dataPath])
        PlotTypeList.extend([chartType])
        xList.extend([inputDict.get('x', [''])])
        yList.extend([inputDict.get('y', [''])])
        hueList.extend([inputDict.get('hue', [''])])
        CategoryList.extend([inputDict.get('category', [''])])
        confidence_intervalList.extend([inputDict.get('confidence_interval', [''])])
        lowessList.extend([inputDict.get('lowess', [''])])
        lineWidthList.extend([inputDict.get('lineWidth', [''])])
        timeVariantColList.extend([inputDict.get('timeVariantCol', [''])])
        timeColList.extend([inputDict.get('timeCol', [''])])
        yearColumn1List.extend([inputDict.get('yearColumn1', [''])])
        yearColumn2List.extend([inputDict.get('yearColumn2', [''])])
        timeVariantCol1List.extend([inputDict.get('timeVariantCol1', [''])])
        timeVariantCol2List.extend([inputDict.get('timeVariantCol2', [''])])
        categoryColList.extend([inputDict.get('categoryCol', [''])])
        categoryListList.extend([inputDict.get('categoryList', [''])])
        categoryVariantColList.extend([inputDict.get('categoryVariantCol', [''])])
        dateColumnList.extend([inputDict.get('dateColumn', [''])])
        timeVariantColumnsList.extend([inputDict.get('timeVariantColumns', [''])])
        compressedDataPathList.extend(compressedPath)


        if(chartType=='barPlot'):
            bar.barPlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], x=inputDict['x'], y=inputDict['y'], hue=inputDict['hue'])
        elif(chartType=='boxenPlot'):
            boxen.boxenPlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], x=inputDict['x'], y=inputDict['y'], hue=inputDict['hue'])
        elif (chartType == 'box'):
            box.boxPlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], x=inputDict['x'], y=inputDict['y'],hue=inputDict['hue'])
        elif (chartType == 'cat'):
            cat.catPlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], x=inputDict['x'], y=inputDict['y'],hue=inputDict['hue'])
        elif (chartType == 'cluster'):
            cluster.clusterMapper(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'])
        elif (chartType == 'count'):
            count.countPlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], category=inputDict['category'])
        elif (chartType == 'dataDist'):
            dataDist.distributionPlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], size=inputDict['size'], category=inputDict['category'], hue=inputDict['hue'])
        elif (chartType == 'dist'):
            dist.distPlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], x=inputDict['x'])
        elif (chartType == 'heat'):
            heat.heatMapper(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'])
        elif (chartType == 'joint'):
            joint.jointPlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], x=inputDict['x'], y=inputDict['y'], type=inputDict['type'])
        elif (chartType == 'kde'):
            kde.kdePlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], x=inputDict['x'], y=inputDict['y'])
        elif (chartType == 'line'):
            line.linePlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], x=inputDict['x'], y=inputDict['y'], hue=inputDict['hue'])
        elif (chartType == 'lm'):
            lm.lmPlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], x=inputDict['x'], y=inputDict['y'], hue=inputDict['hue'])
        elif (chartType == 'pairPlot'):
            pair.pairPlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], size=inputDict['size'], hue=inputDict['hue'])
        elif (chartType == 'pairGrid'):
            pairGrid.pairGridPlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], hue=inputDict['hue'])
        elif (chartType == 'point'):
            point.pointPlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], x=inputDict['x'], y=inputDict['y'], hue=inputDict['hue'])
        elif (chartType == 'reg'):
            reg.regPlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], x=inputDict['x'], y=inputDict['y'], confidence_interval=inputDict['confidence_interval'])
        elif (chartType == 'relation'):
            relation.relPlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], x=inputDict['x'], y=inputDict['y'], hue=inputDict['hue'])
        elif (chartType == 'resid'):
            resid.residPlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], x=inputDict['x'], y=inputDict['y'], lowess=inputDict['lowess'])
        elif (chartType == 'rug'):
            rug.rugPlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], x=inputDict['x'])
        elif (chartType == 'scatter'):
            scatter.scatterPlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], x=inputDict['x'], y=inputDict['y'], hue=inputDict['hue'])
        elif (chartType == 'strip'):
            strip.stripPlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], x=inputDict['x'], y=inputDict['y'], lineWidth=inputDict['lineWidth'])
        elif (chartType == 'swarm'):
            swarm.swarmPlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], x=inputDict['x'], y=inputDict['y'], hue=inputDict['hue'])
        elif (chartType == 'violin'):
            violin.violinPlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], x=inputDict['x'], y=inputDict['y'], hue=inputDict['hue'])
        elif (chartType == 'scatterEncircle'):
            scatterEncircle.scatterWithEncirclePlotter()
            violin.violinPlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], x=inputDict['x'], y=inputDict['y'], hue=inputDict['hue'])
        elif (chartType == 'divBars'):
            divBars.divergingBarsPlotter(df=data,outputPath=outputPath,x=inputDict['x'], y=inputDict['y'],description='Diverging Bars ',ticketId=inputDict['ticketId'])
        elif (chartType == 'divBarText'):
            divBarText.divergingBarTextPlotter(df=data,outputPath=outputPath,x=inputDict['x'], y=inputDict['y'],description='Diverging Bars with Text ',ticketId=inputDict['ticketId'])
        elif (chartType == 'divLolly'):
            divLolly.divergingLollipopPlotter(df=data,outputPath=outputPath,x=inputDict['x'], y=inputDict['y'],description='Diverging Lollipop ',ticketId=inputDict['ticketId'])
        elif (chartType == 'divDot'):
            divDot.divergingDotPlotter(df=data,outputPath=outputPath,x=inputDict['x'], y=inputDict['y'],description='Diverging Dots ',ticketId=inputDict['ticketId'])
        elif (chartType == 'area'):
            area.areaChartPlotter(df=data,outputPath=outputPath,timeCol=inputDict['timeCol'],timeVariantCol=inputDict['timeVariantCol'],description=' Area chart',ticketId=inputDict['ticketId'])
        elif (chartType == 'slope'):
            slope.slopePlotter(df=data,yearColumn1=inputDict['yearColumn1'],yearColumn2=inputDict['yearColumn2'],timeVariantCol=inputDict['timeVariantCol'],description='slope chart',outputPath=outputPath,ticketId=inputDict['ticketId'])
        elif (chartType == 'dumbell'):
            dumbell.dumbellChartPlotter(df=data,timeVariantCol1=inputDict['timeVariantCol1'],timeVariantCol2=inputDict['timeVariantCol2'],description=' dumbell chart',outputPath=outputPath,ticketId=inputDict['ticketId'])
        elif (chartType == 'densityCurve'):
            densityCurve.denCurveWithHistoPlotter(df=data,categoryCol=inputDict['categoryCol'],categoryList=inputDict['categoryList'],categoryVariantCol=inputDict['categoryVariantCol'],description='Density Plot with Histogram',outputPath=outputPath,ticketId=inputDict['ticketId'])
        elif (chartType == 'distDot'):
            distDot.distributedDotPlot(df_raw=data,x=inputDict['x'],y=inputDict['y'],category=inputDict['category'],categoryList=inputDict['categoryList'],description='Distributed dot plot',outputPath=outputPath,ticketId=inputDict['ticketId'])
        elif (chartType == 'density'):
            density.densityPlotter(df=data,categoryCol=inputDict['categoryCol'],categoryList=inputDict['categoryList'],categoryVariantCol=inputDict['categoryVariantCol'],description='Density Plot',outputPath=outputPath,ticketId=inputDict['ticketId'])
        elif (chartType == 'dotBox'):
            dotBox.dotBoxPlotter(df=data,outputPath=outputPath,ticketId=inputDict['ticketId'], x=inputDict['x'], y=inputDict['y'],hue=inputDict['hue'],description='Dot + Box Plot')
        elif (chartType == 'tufte'):
            tufte.tufteBoxPlotter(dataframe=data,outputPath=outputPath,ticketId=inputDict['ticketId'], x=inputDict['x'])
        elif (chartType == 'peakTrough'):
            peakTrough.peakTroughTimeSeriesPlotter(df=data,dateColumn=inputDict['dateColumn'],timeVariantCol=inputDict['timeVariantCol'],description='time series analysis',outputPath=outputPath,ticketId=inputDict['ticketId'])
        elif (chartType == 'cross'):
            cross.crossCorelationPlotter(dataframe=data,x=inputDict['x'],y=inputDict['y'],title='Cross correlation Plot',outputPath=outputPath,ticketId=inputDict['ticketId'])
        elif (chartType == 'decomp'):
            decompdataPath.timeSeriesDecompositionPlotter(df=data,dateColumn=inputDict['dateColumn'],timeVariantCol=inputDict['timeVariantCol'],description='time series analysis',outputPath=outputPath,ticketId=inputDict['ticketId'])
        elif (chartType == 'multiple'):
            multiple.multipleTimeSeriesPlotter(df=data,dateColumn=inputDict['dateColumn'],timeVariantCols=inputDict['timeVariantCols'],description='multiple time series',outputPath=outputPath,ticketId=inputDict['ticketId'])
        elif (chartType == 'seasonal'):
            seasonal.seasonalPlotter(df=data,dateColumn=inputDict['dateColumn'],timeVariantCol=inputDict['timeVariantCol'],description='Monthly seasonal plot',outputPath=outputPath,ticketId=inputDict['ticketId'])

    visual_data['TicketId'] = TicketIdList
    visual_data['Data'] = DataList
    visual_data['CompressedDataPath'] = compressedDataPathList
    visual_data['PlotType'] = PlotTypeList
    visual_data['x'] = xList
    visual_data['y'] = yList
    visual_data['hue'] = hueList
    visual_data['Category'] = CategoryList
    visual_data['confidence_interval'] = confidence_intervalList
    visual_data['lowess'] = lowessList
    visual_data['lineWidth'] = lineWidthList
    visual_data['timeVariantCol'] = timeVariantColList
    visual_data['timeCol'] = timeColList
    visual_data['yearColumn1'] = yearColumn1List
    visual_data['yearColumn2'] = yearColumn2List
    visual_data['timeVariantCol1'] = timeVariantCol1List
    visual_data['timeVariantCol2'] = timeVariantCol2List
    visual_data['categoryCol'] = categoryColList
    visual_data['categoryList'] = categoryListList
    visual_data['categoryVariantCol'] = categoryVariantColList
    visual_data['dateColumn'] = dateColumnList
    visual_data['timeVariantColumns'] = timeVariantColumnsList



    visual_data.to_excel(outputPath+r'\visualisation_metadata_' + inputDict['ticketId'] +'_'+dataType+ r'.xlsx')


if __name__ == '__main__':
    print('started main')
    dataPath = r"D:\Personal\SmartIT\data"
    data = pd.read_excel(r"D:\Personal\SmartIT\data\diabetes.xlsx")
    attDict = {'ticketId':'123456','x':"BMI", 'y':"Outcome", 'hue':"Pregnancies",'confidence_interval':70}
    chartList = ['barPlot','boxenPlot','divLolly','reg']
    print('chart plotter called')
    # dataPath, data, chartsList, inputDict, splitPercentage, pcaComp = 0
    chartPlotter(dataPath,data,chartList,attDict,10)

    X_train, X_test = train_test_split(data, test_size=0.10, random_state=42)
    X_test.to_excel(r"D:\Personal\SmartIT\data\percenteage.xlsx")


