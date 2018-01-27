package org.rnn.lstm;

import java.io.IOException;
import java.util.List;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.WindowConstants;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.RefineryUtilities;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.rnn.lstm.preprocessing.Preprocessing;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;




public class MultiTimestepAnomalyDetection {

	private static final Logger LOGGER = LoggerFactory.getLogger(MultiTimestepAnomalyDetection.class);
	private static final String temperatureFilePath = "E:\\Thesis\\Datasets\\TemporalDatasets\\tempData.csv";
	
	
	public static void main(String [] args) throws IOException, InterruptedException {
	
//		Preprocessing.splitTrainandTest(temperatureFilePath);
		List<String> rawStrings = Preprocessing.getData();
		int skipRow = 0;
		String delimiter = ",";
		int miniBatchSize = 10;
		int numOfVariables = 3;
		
		int trainSize = 9700;
		int testSize = 2200;
		
		SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(skipRow, delimiter );
		trainFeatures.initialize(new NumberedFileInputSplit(Preprocessing.featuresDirTrain.getAbsolutePath() + "/train_%d.csv", 0, trainSize-1));
		
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader(skipRow, delimiter);
        trainLabels.initialize(new NumberedFileInputSplit(Preprocessing.labelsDirTrain.getAbsolutePath() + "/train_%d.csv", 0, trainSize-1));

        DataSetIterator trainDataIter = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, -1, true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
        testFeatures.initialize(new NumberedFileInputSplit(Preprocessing.featuresDirTest.getAbsolutePath() + "/test_%d.csv", trainSize, trainSize + testSize - 1));
        SequenceRecordReader testLabels = new CSVSequenceRecordReader();
        testLabels.initialize(new NumberedFileInputSplit(Preprocessing.labelsDirTest.getAbsolutePath() + "/test_%d.csv", trainSize, trainSize + testSize - 1));

        DataSetIterator testDataIter = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, -1, true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        //Normalize the training data
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
        normalizer.fitLabel(true);
        normalizer.fit(trainDataIter);              //Collect training data statistics
        trainDataIter.reset();
        
        trainDataIter.setPreProcessor(normalizer);
        testDataIter.setPreProcessor(normalizer);
        
        DataSet ds = trainDataIter.next();
        System.out.println(ds.get(9));
        
        
     // ----- Configure the network -----
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(140)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS)
            .learningRate(0.001)
            .list()
            .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(numOfVariables).nOut(5)
                .build())
            .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY).nIn(5).nOut(1).build())
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
//        net.setListeners(new ScoreIterationListener(1));
        
//      //Initialize the user interface backend
//      UIServer uiServer = UIServer.getInstance();
//
//      //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
//      StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later
//      
//      //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
//      uiServer.attach(statsStorage);
//
//      //Then add the StatsListener to collect this information from the network, as it trains
//      net.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1));
//        
//      int nEpochs = 3;
//
//      for (int i = 0; i < nEpochs; i++) {
//        net.fit(trainDataIter);
//        trainDataIter.reset();
//        LOGGER.info("Epoch " + i + " complete. Time series evaluation:");
//
//        RegressionEvaluation evaluation = new RegressionEvaluation(1);
//
//        //Run evaluation. This is on 25k reviews, so can take some time
//        while (testDataIter.hasNext()) {
//            DataSet t = testDataIter.next();
//            INDArray features = t.getFeatureMatrix();
//            INDArray labels = t.getLabels();
//            INDArray predicted = net.output(features, true);
//
//            evaluation.evalTimeSeries(labels, predicted);
//
//        }
//        testDataIter.reset();
//        System.out.println(evaluation.stats());
//      }
//        
//        //Init rnnTimeStep with train data and predict test data
//      while (trainDataIter.hasNext()) {
//            DataSet t = trainDataIter.next();
//            net.rnnTimeStep(t.getFeatureMatrix());
//      }
//        trainDataIter.reset();
//
//        while (testDataIter.hasNext()) {
//		    DataSet t = testDataIter.next();
//		    INDArray predicted = net.rnnTimeStep(t.getFeatureMatrix());
//		    normalizer.revertLabels(predicted);
//		    System.out.println(predicted);
//        }
//        
//        //Convert raw string data to IndArrays for plotting
//        INDArray trainArray = createIndArrayFromStringList(rawStrings, 0, trainSize);
//        INDArray testArray = createIndArrayFromStringList(rawStrings, trainSize, testSize);
//
//        //Create plot with out data
//        XYSeriesCollection c = new XYSeriesCollection();
////        createSeries(c, trainArray, 0, "Train data");
//        createTrainSeries(c, testArray, 0, "Actual test data");
//        createTestSeries(c, testDataIter, 0, "Predicted test data", normalizer, net);
//        plotDataset(c);
//
//        LOGGER.info("----- Example Complete -----");
//    }
//
//    private static void createTestSeries(XYSeriesCollection c, DataSetIterator testDataIter, int i, String string, NormalizerMinMaxScaler normalizer, MultiLayerNetwork net) {
//    	
//    	while (testDataIter.hasNext()) {
//    		
//		    DataSet t = testDataIter.next();
//		    INDArray predicted = net.rnnTimeStep(t.getFeatureMatrix());
//		    normalizer.revertLabels(predicted);
//		    
//        }
//		
//	}
//
//	/**
//     * Creates an IndArray from a list of strings
//     * Used for plotting purposes
//     */
//    private static INDArray createIndArrayFromStringList(List<String> rawStrings, int startIndex, int length) {
//        List<String> stringList = rawStrings.subList(startIndex, startIndex + length);
//
//        double[][] primitives = new double[3][stringList.size()];
//        for (int i = 0; i < stringList.size(); i++) {
//            String[] vals = stringList.get(i).split(",");
//            for (int j = 0; j < vals.length; j++) {
//                primitives[j][i] = Double.valueOf(vals[j]);
//            }
//        }
//        return Nd4j.create(new int[]{1, length}, primitives);
//    }
//
//    /**
//     * Used to create the different time series for plotting purposes
//     */
//    private static void createTrainSeries(XYSeriesCollection seriesCollection, INDArray data, int offset, String name) {
//        int nRows = data.shape()[2];
//        boolean predicted = name.startsWith("Predicted");
//        XYSeries series = new XYSeries(name);
//        for (int i = 0; i < nRows; i++) {
//            if (predicted)
//                series.add(i + offset, data.slice(0).getDouble(i));
//            else
//                series.add(i + offset, data.slice(0).getDouble(i));
//        }
//        seriesCollection.addSeries(series);
//    }
//
//    /**
//     * Generate an xy plot of the datasets provided.
//     */
//    private static void plotDataset(XYSeriesCollection c) {
//
//        String title = "Regression example";
//        String xAxisLabel = "Timestep";
//        String yAxisLabel = "Number of passengers";
//        PlotOrientation orientation = PlotOrientation.VERTICAL;
//        boolean legend = true;
//        boolean tooltips = false;
//        boolean urls = false;
//        JFreeChart chart = ChartFactory.createXYLineChart(title, xAxisLabel, yAxisLabel, c, orientation, legend, tooltips, urls);
//
//        // get a reference to the plot for further customization...
//        final XYPlot plot = chart.getXYPlot();
//
//        // Auto zoom to fit time series in initial window
//        final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
//        rangeAxis.setAutoRange(true);
//        
//        JPanel panel = new ChartPanel(chart);
//
//        JFrame f = new JFrame();
//        f.add(panel);
//        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
//        f.pack();
//        f.setTitle("Training Data");
//
//        RefineryUtilities.centerFrameOnScreen(f);
//        f.setVisible(true);
//        
	}
}
