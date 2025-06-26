# 📄 Bulk Payment Tool Documentation

## ❓ Purpose

The **Bulk Payment Tool** is designed to assist users in filtering outstanding entries and creating payment entries automatically, streamlining the payment process.

## 🌐 Scope

This document outlines the steps and prerequisites for using the Bulk Payment Tool effectively.

## 🔑 Prerequisites

- Access to the Bulk Payment Tool Doctype is required.

---

## 📋 Standard Operating Procedure (SOP)

The following guidelines define a consistent and structured method for performing tasks within the Bulk Payment Tool. This ensures clarity, efficiency, and accountability during its use:

- **Accessibility**: Confirm that you have access to the Bulk Payment Tool before proceeding.
- **Filter Application**: Use the provided filters to narrow down entries, focusing on company, branch, supplier, or payment details as needed.
- **Accuracy**: Double-check all inputs such as dates, amounts, and payment methods before saving or submitting documents.
- **Process Verification**: Verify the populated entries in the Items table after using the "Get Outstanding" button to ensure all required data is included.
- **Submission**: Only submit documents after entering correct references and confirming payment details.

> Adhering to these steps ensures the seamless execution of payments and reduces the risk of errors.

---

## 🧭 Step-by-Step Instructions

### Step 1: Access the Bulk Payment Tool

- Navigate to the **Bulk Payment Tool** module on the Frappe dashboard or search **Bulk Payment Tool** in the Awesome Bar.
- Click on **Add Bulk Payment Tool** to open the Bulk Payment interface.

### Step 2: Set Filters to Retrieve Outstanding Entries

Within the Bulk Payment Tool, users will see a set of filters to define specific criteria for retrieving entries. The available filters include:

- Company  
- Branch  
- Supplier Group  
- Supplier  
- As On Date  
- Payment Posting Date  
- Mode of Payment  
- Company Bank Account  
- **Amount to Disburse**: Specify the total amount to disburse (e.g., ₹500,000).

### Step 3: Save the Document

- After setting all the filters, click **Save** to preserve the document.

### Step 4: Retrieve Outstanding Entries

- Once the document is saved, click the **Get Outstanding** button.
- This action populates the **Items** table with all outstanding entries.

### Step 5: Enter Amount to Pay

- In the **Items** table, locate the **To Pay** column.
- Specify the amount to disburse for each entry.

### Step 6: Process Payments

- After entering amounts in the **To Pay** column, click the **Process Payments** button.
- The system consolidates multiple entries for the same party into a single row in the **Payment** table.

### Step 7: Enter Payment References

For each row in the **Payment** table:

- Enter the **Reference Number**
- Provide the **Reference Date**

### Step 8: Submit the Document

- Click the **Submit** button to finalize the process.
- **Payment entries** are automatically created for each row in the Payment table.

---

## 📜 Notes

- Ensure that all details, such as reference numbers and dates, are accurate before submitting.
- Use the filters effectively to retrieve only the required outstanding entries.

