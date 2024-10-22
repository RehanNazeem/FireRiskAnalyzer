import SwiftUI
import CoreML
import Vision
import UIKit
import CoreVideo

// function to resize the image
extension UIImage {
    func resized(to targetSize: CGSize) -> UIImage? {
        let size = self.size
        
        let widthRatio  = targetSize.width  / size.width
        let heightRatio = targetSize.height / size.height
        
        // scale factor
        let scaleFactor = min(widthRatio, heightRatio)
        
        // Calculate the new size
        let newSize = CGSize(width: size.width * scaleFactor, height: size.height * scaleFactor)
        
        // Create a graphics context
        UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
        self.draw(in: CGRect(origin: .zero, size: newSize))
        
        // Get the resized image
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return resizedImage
    }

    // convert UIImage to CVPixelBuffer
    func pixelBuffer() -> CVPixelBuffer? {
        let attributes: [NSObject: AnyObject] = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ]
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         Int(self.size.width),
                                         Int(self.size.height),
                                         kCVPixelFormatType_32ARGB,
                                         attributes as CFDictionary,
                                         &pixelBuffer)
        
        guard (status == kCVReturnSuccess) else {
            print("Error creating pixel buffer: \(status)")
            return nil
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer!, [])
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData,
                                width: Int(self.size.width),
                                height: Int(self.size.height),
                                bitsPerComponent: 8,
                                bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!),
                                space: rgbColorSpace,
                                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        context?.draw(self.cgImage!, in: CGRect(x: 0, y: 0, width: self.size.width, height: self.size.height))
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, [])
        
        return pixelBuffer
    }
}


struct ContentView: View {
    @State private var image: UIImage?
    @State private var riskLevel: String = ""
    @State private var recommendations: String = ""
    @State private var showCameraPicker = false
    @State private var showGalleryPicker = false
    @State private var selectedImage: UIImage?

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 20)
                .strokeBorder(Color.orange, lineWidth: 5)
                .background(Color.white)
                .edgesIgnoringSafeArea(.all)
            
            VStack(spacing: 8) { //spacing adjustment for elements in vstack
                VStack(spacing: 4) { //spacing adjustment for elements in this vstack
                    Text("Terrain Fire Risk Analyzer")
                        .font(.system(size: 25))
                        .foregroundColor(.orange)
                        .bold()
                    
                    Image("mainscreenlogo")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 250, height: 60)
                }
                .padding(.bottom, 8) //reduce padding below this vstack
                
                VStack(spacing: 8) { //adjust spacing betweemn image, button and text
                    if let selectedImage = selectedImage {
                        Image(uiImage: selectedImage)
                            .resizable()
                            .scaledToFit()
                            .frame(width: 200, height: 200)
                    }

                    Button("Select Image from Gallery") {
                        // reset both state for camera and gallery
                        showCameraPicker = false
                        showGalleryPicker = true
                    }
                    .padding(.bottom, 4) // Reduce the padding below the button

                    Button("Take Photo") {
                        if UIImagePickerController.isSourceTypeAvailable(.camera) {
                            // Reset both states before setting the camera state
                            showGalleryPicker = false
                            showCameraPicker = true
                        } else {
                            print("Camera is not available on this device.")
                        }
                    }
                    .padding(.bottom, 8) //reduce padding below button
                }
                
                HStack {
                    Text("Fire Risk Level: ")
                        .bold() //first part bold
                    +
                    Text("\(riskLevel)") //second part normal
                }
                .padding(.bottom, 8) //reduce padding below button
                
                HStack {
                    Text("Recommendations: ")
                        .bold() //firt part of text bold
                    +
                    Text("\(recommendations)") //keep second part normal
                }
                .padding(.bottom, 4) //padding
                
                
            }
            .padding(.horizontal, 16) //padding
        }
        .sheet(isPresented: $showGalleryPicker) {
            ImagePicker(selectedImage: $selectedImage, sourceType: .photoLibrary)
                .onDisappear {
                    if let selectedImage = selectedImage {
                        analyzeImage(image: selectedImage)
                    }
                }
        }
        .sheet(isPresented: $showCameraPicker) {
            ImagePicker(selectedImage: $selectedImage, sourceType: .camera)
                .onDisappear {
                    if let selectedImage = selectedImage {
                        analyzeImage(image: selectedImage)
                    }
                }
        }
    }


    //analyse the image using new model
    func analyzeImage(image: UIImage) {
        //resize the image to 224 x 224, this is the expected size by FireriskimageML model
        let targetSize = CGSize(width: 224, height: 224)
        guard let resizedImage = image.resized(to: targetSize),
              let pixelBuffer = resizedImage.pixelBuffer() else {
            print("Failed to resize or convert image to pixel buffer.")
            return
        }

        //Load the FireriskimageML model
        guard let model = try? FireRiskImageML(configuration: MLModelConfiguration()) else {
            print("Failed to load FireRiskImageML model.")
            return
        }

        //create a VNCoreMLModel from the FireriskimageML model
        guard let visionModel = try? VNCoreMLModel(for: model.model) else {
            print("Failed to create VNCoreMLModel.")
            return
        }

        // request to analyze the image
        let request = VNCoreMLRequest(model: visionModel) { request, error in
            if let error = error {
                print("Error during request: \(error.localizedDescription)")
                return
            }

            if let results = request.results as? [VNClassificationObservation], let topResult = results.first {
                DispatchQueue.main.async {
                    self.riskLevel = "\(topResult.identifier) - \(Int(topResult.confidence * 100))% confidence"
                    self.recommendations = self.getRecommendations(for: topResult.identifier)
                }
            } else {
                print("No results found.")
            }
        }

        //perform the request VNImageRequestHandler
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        do {
            try handler.perform([request])
        } catch {
            print("Error performing request: \(error.localizedDescription)")
        }
    }

    
    
    // Analyze the image using the MobileNetV2 2 model - this was the intial integration code
    //using existing ML model . it is commented since a custom model was created.
    /*func analyzeImage(image: UIImage) {
        let targetSize = CGSize(width: 224, height: 224)
        guard let resizedImage = image.resized(to: targetSize),
              let pixelBuffer = resizedImage.pixelBuffer() else {
            print("Failed to resize or convert image to pixel buffer.")
            return
        }

        //load MobileNetV2 model
        guard let model = try? MobileNetV2_2(configuration: MLModelConfiguration()) else {
            print("Failed to load MobileNetV2 2 model.")
            return
        }
        //VNCoreMLModel from VNCoreMLModel model

        guard let visionModel = try? VNCoreMLModel(for: model.model) else {
            print("Failed to create VNCoreMLModel.")
            return
        }

        //reqeust to analyse image
        let request = VNCoreMLRequest(model: visionModel) { request, error in
            if let error = error {
                print("Error during request: \(error.localizedDescription)")
                return
            }

            if let results = request.results as? [VNClassificationObservation], let topResult = results.first {
                DispatchQueue.main.async {
                    self.riskLevel = "\(topResult.identifier) - \(Int(topResult.confidence * 100))% confidence"
                    self.recommendations = self.getRecommendations(for: topResult.identifier)
                }
            } else {
                print("No results found.")
            }
        }

        // reqeust using VNImageRequestHandler
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        do {
            try handler.perform([request])
        } catch {
            print("Error performing request: \(error.localizedDescription)")
        }
    }
*///end of mobilenet V2 2
    
    //Recommendation based on risk level
    func getRecommendations(for risk: String) -> String {
        switch risk {
        case "Highrisk":
            return "Take immediate action! Consider controlled burn, Maintain 30-100 ft of clear space around boundary, Create barrier to contain fire, Clear flammable vegetation and debris. Additionally, Educate community on fire safety to enhance preparedness and response!"
        case "Mediumrisk":
            return "Be on Alert! Consider regular vegetation trimming, Controlled burning, Educate community  and Emergency planning to mitigate fire risks."
        case "Lowrisk":
            return "Maintain basic precautions, Keep flammable materials away from heat sources, Educate community to minimize potential fire risks."
        default:
            return "No recommendations available."
        }
    }
}
//Imagepicker to allow selecting images from the gallery or taking a photo
struct ImagePicker: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?
    @Environment(\.presentationMode) var presentationMode
    var sourceType: UIImagePickerController.SourceType

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = sourceType
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        return Coordinator(self)
    }

    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        let parent: ImagePicker

        init(_ parent: ImagePicker) {
            self.parent = parent
        }

        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let image = info[.originalImage] as? UIImage {
                parent.selectedImage = image
            }
            parent.presentationMode.wrappedValue.dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
}
