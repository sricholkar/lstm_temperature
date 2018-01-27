package org.rnn.lstm.preprocessing;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.io.FileUtils;
import org.datavec.api.util.ClassPathResource;
import org.rnn.lstm.pojo.TempPOJO;

import au.com.bytecode.opencsv.CSVReader;


public class Preprocessing {

	private static final String baseDirPath = "/temperature";
	public static File baseDir = initBaseFile(baseDirPath);
	public static File baseTrainDir = new File(baseDir, "multiTimestepTrain");
    public static File featuresDirTrain = new File(baseTrainDir, "features");
    public static File labelsDirTrain = new File(baseTrainDir, "labels");
    public static File baseTestDir = new File(baseDir, "multiTimestepTest");
    public static File featuresDirTest = new File(baseTestDir, "features");
    public static File labelsDirTest = new File(baseTestDir, "labels");
    
    public static List<String> getData() throws IOException {
    	Path rawPath = Paths.get(baseDir.getAbsolutePath() + "/tempData.csv");    	
    	List<String> rawStrings = Files.readAllLines(rawPath);
    	return rawStrings;
    }
    
	public static void splitTrainandTest(String temperatureFilePath) throws IOException {
		
		File baseDir = initBaseFile(baseDirPath);
		File baseTrainDir = new File(baseDir, "multiTimestepTrain");
	    File featuresDirTrain = new File(baseTrainDir, "features");
	    File labelsDirTrain = new File(baseTrainDir, "labels");
	    File baseTestDir = new File(baseDir, "multiTimestepTest");
	    File featuresDirTest = new File(baseTestDir, "features");
	    File labelsDirTest = new File(baseTestDir, "labels");
	    
	    List<TempPOJO> tempList = new ArrayList<TempPOJO>();
		int trainSize = 9700;
		int testSize = 2200;
		int numOfTimesteps = 20;
		File temperatureFile = new File(temperatureFilePath);
		Reader CSVReader;
		Iterable<CSVRecord> records = null;
		
		try {						//read CSV file containing the dataset
			CSVReader = new FileReader(temperatureFile);
			records = CSVFormat.EXCEL.withHeader().parse(CSVReader);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			
			e.printStackTrace();
		} 
		 //object created to iterate through CSV file
		StringBuilder sb = new StringBuilder();
		int i = 0;
		for (final CSVRecord record: records) //storing the feature values in a list of type WeatherPOJ 
		{
			i++;
			final Integer count = new Integer(i);
			tempList.add(new TempPOJO() {{
				this.id = count;
				this.date = Integer.valueOf(String.valueOf(record.get("DATE")));
				this.actual_avg_temperature = Double.valueOf(String.valueOf(record.get("TAVG")));
				this.actual_tmax_temp = Double.valueOf(String.valueOf(record.get("TMAX")));
				this.actual_tmin_temp = Double.valueOf(String.valueOf(record.get("TMIN")));
				
			}});
		}
		
		for (TempPOJO weather: tempList) {
			if (weather.id <= 12100)
			sb.append(weather.actual_avg_temperature).append(", ").append(weather.actual_tmax_temp).append(", ").append(weather.actual_tmin_temp).append("\n");
		}
		OutputStreamWriter fw = new OutputStreamWriter(new FileOutputStream(baseDir.getAbsolutePath() + "/tempData.csv"));
		fw.write(sb.toString());
		fw.flush();
		fw.close();
		
		//Remove all files before generating new ones
        FileUtils.cleanDirectory(featuresDirTrain);
        FileUtils.cleanDirectory(labelsDirTrain);
        FileUtils.cleanDirectory(featuresDirTest);
        FileUtils.cleanDirectory(labelsDirTest);
        
		Path rawPath = Paths.get(baseDir.getAbsolutePath() + "/tempData.csv");
		List<String> rawStrings = Files.readAllLines(rawPath, Charset.defaultCharset());
	    setNumOfVariables(rawStrings);
	    
	    for (int j = 0; j < trainSize; j++) {
            Path featuresPath = Paths.get(featuresDirTrain.getAbsolutePath() + "/train_" + j + ".csv");
            Path labelsPath = Paths.get(labelsDirTrain + "/train_" + j + ".csv");
            for (int k = 0; k < numOfTimesteps; k++) {
                Files.write(featuresPath, rawStrings.get(j + k).concat(System.lineSeparator()).getBytes(), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
            }
            Files.write(labelsPath, rawStrings.get(j + numOfTimesteps).substring(0, 2).getBytes(), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
        }

        for (int j = trainSize; j < testSize + trainSize; j++) {
            Path featuresPath = Paths.get(featuresDirTest + "/test_" + j + ".csv");
            Path labelsPath = Paths.get(labelsDirTest + "/test_" + j + ".csv");
            for (int k = 0; k < numOfTimesteps; k++) {
                Files.write(featuresPath, rawStrings.get(j + k).concat(System.lineSeparator()).getBytes(), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
            }
            Files.write(labelsPath, rawStrings.get(j + numOfTimesteps).substring(0, 2).getBytes(), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
        }
	}

	private static void setNumOfVariables(List<String> rawStrings) {
        int numOfVariables = rawStrings.get(0).split(",").length;
    }

	private static File initBaseFile(String baseDirPath) {
        try {
            return new ClassPathResource(baseDirPath).getFile();
        } catch (IOException e) {
            throw new Error(e);
        }
    }
}
